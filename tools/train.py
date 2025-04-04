# mypy: ignore-errors
from __future__ import annotations

from random import random
from tap import Tap
from typing import TYPE_CHECKING

import cv2

from codecarbon import OfflineEmissionsTracker
from ultralytics import YOLO
from ultralytics.data.augment import (
    Albumentations,
    Compose,
    CopyPaste,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import de_parallel

import wandb

if TYPE_CHECKING:
    from typing import Any

    from ultralytics.data.augment import Compose
    from ultralytics.data.base import BaseDataset
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils import IterableSimpleNamespace

tracker: OfflineEmissionsTracker | None = None


class GaussianBlur:
    def __init__(self, p: float = 0.5, kernel_size: tuple[int, int] = (3, 3), sigma: float = 0.0) -> None:
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert all(v % 2 == 1 and v > 0 for v in kernel_size), "The kernel size should be positive and odd."

        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random() > self.p:
            labels["img"] = cv2.GaussianBlur(labels["img"], self.kernel_size, self.sigma)

        return labels


class CustomDatasetLoader(YOLODataset):
    def v8_transforms_custom(dataset: BaseDataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False) -> Compose:
        mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
        affine = RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )

        pre_transform = Compose([mosaic, affine])
        if hyp.copy_paste_mode == "flip":
            pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
        else:
            pre_transform.append(
                CopyPaste(
                    dataset,
                    pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                    p=hyp.copy_paste,
                    mode=hyp.copy_paste_mode,
                )
            )
        flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
        if dataset.use_keypoints:
            kpt_shape = dataset.data.get("kpt_shape", None)
            if len(flip_idx) == 0 and hyp.fliplr > 0.0:
                hyp.fliplr = 0.0
                LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
            elif flip_idx and (len(flip_idx) != kpt_shape[0]):
                raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

        return Compose(
            [
                GaussianBlur(0.5, (5, 5)),  # Added GaussianBlur
                pre_transform,
                MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
                Albumentations(p=1.0),
                RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                RandomFlip(direction="vertical", p=hyp.flipud),
                RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
            ]
        )

    def build_transforms(self, hyp: IterableSimpleNamespace | None = None) -> Compose:
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0

            transforms = self.v8_transforms_custom(self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms


class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> BaseDataset:
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)

        return CustomDatasetLoader(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


def start_track_emissions(trainer: BaseTrainer) -> None:
    global tracker

    # Retrieve stuff from WandB
    if wandb.run is not None:
        tracker = OfflineEmissionsTracker(
            country_iso_code="NLD",
            project_name=f"{wandb.run.project_name()}/{wandb.run.id}",
            experiment_id=wandb.run.id,
            experiment_name=wandb.run.name,
            log_level="warning",
        )
    else:
        tracker = OfflineEmissionsTracker(country_iso_code="NLD", log_level="error")

    tracker.start()


def stop_track_emissions(trainer: BaseTrainer) -> None:
    global tracker

    try:
        if tracker is not None:
            tracker.stop()

            if wandb.run is not None:
                if "energy_consumed" not in wandb.config:
                    wandb.config["energy_consumed"] = tracker.final_emissions_data.energy_consumed  # KwH

                if "total_emmissions" not in wandb.config:
                    wandb.config["total_emissions"] = tracker.final_emissions_data.emissions  # CO2 eqv

            LOGGER.info(f"Codecarbon: used {tracker.final_emissions_data.energy_consumed} KwH energy")
            LOGGER.info(f"Codecarbon: emitted {tracker.final_emissions_data.emissions} kg CO2 eqv")
    except Exception:
        pass


def parse_input(input: str) -> str | int | float | bool:
    try:
        return int(input)
    except ValueError:
        try:
            return float(input)
        except ValueError:
            if input.lower() == "false" or input.lower() == "true":
                return bool(input)
            return input


class ArgumentParser(Tap):
    name: str  # Name of training
    yolo_args: dict[str, str | int | float | bool]  # Yolo arguments for training

    def configure(self) -> None:
        self.add_argument("name")
        self.add_argument("yolo_args", type=str, nargs="*", metavar="NAME=VAR")

    def process_args(self) -> None:
        self.yolo_args = {k: v if v.startswith("/") else parse_input(v) for k, v in (arg.split("=") for arg in self.yolo_args)}


def main() -> None:
    args = ArgumentParser().parse_args()

    model_name = args.yolo_args.get("model", "yolov8n.pt")

    network = YOLO(model=model_name)
    network.add_callback("on_train_start", start_track_emissions)
    network.add_callback("on_train_end", stop_track_emissions)
    network.train(trainer=CustomTrainer, name=args.name, **args.yolo_args)


if __name__ == "__main__":
    main()
