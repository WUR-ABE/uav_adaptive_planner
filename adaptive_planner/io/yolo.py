from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from yaml import safe_dump, safe_load

import numpy as np

from exiftool import ExifToolHelper

from adaptive_planner import setup_logging
from adaptive_planner.io import get_image_paths
from adaptive_planner.predict import Detection
from adaptive_planner.utils import IMAGE_EXTENSIONS, Annotation

if TYPE_CHECKING:
    from typing import Any


log = setup_logging(__name__)


@dataclass
class YoloDataset:
    dataset_root_folder: Path
    train_image_folder: Path
    validation_image_folder: Path
    test_image_folder: Path | None
    train_label_folder: Path
    validation_label_folder: Path
    test_label_folder: Path | None
    class_names: list[str]

    @property
    def image_paths(self) -> dict[str, Path | None]:
        return {"train": self.train_image_folder, "val": self.validation_image_folder, "test": self.test_image_folder}

    @property
    def label_paths(self) -> dict[str, Path | None]:
        return {
            "train": self.train_label_folder,
            "val": self.validation_label_folder,
            "test": self.test_label_folder,
        }

    def initialise_dataset(self) -> None:
        self.dataset_root_folder.mkdir(parents=True)
        self.train_image_folder.mkdir(parents=True)
        self.validation_image_folder.mkdir(parents=True)
        self.train_label_folder.mkdir(parents=True)
        self.validation_label_folder.mkdir(parents=True)

        if self.test_image_folder and self.test_label_folder:
            self.test_image_folder.mkdir(parents=True)
            self.test_label_folder.mkdir(parents=True)

        self.write_dataset_file()
        self.write_classes_file()

    def get_train_images(self) -> list[Path]:
        return get_image_paths(self.train_image_folder)

    def get_validation_images(self) -> list[Path]:
        return get_image_paths(self.validation_image_folder)

    def get_test_images(self) -> list[Path]:
        if self.test_image_folder is None:
            return []

        return get_image_paths(self.test_image_folder)

    def get_num_train_images(self) -> int:
        return len(self.get_train_images())

    def get_num_validation_images(self) -> int:
        return len(self.get_validation_images())

    def get_num_test_images(self) -> int:
        return len(self.get_test_images())

    def get_label(self, image_name: str) -> list[Annotation]:
        if image_file := self._filename_in_folder(self.train_image_folder, image_name, image=True):
            label_file = self.train_label_folder / (image_name + ".txt")
        elif image_file := self._filename_in_folder(self.validation_image_folder, image_name, image=True):
            label_file = self.validation_label_folder / (image_name + ".txt")
        elif (
            self.test_image_folder is not None
            and self.test_label_folder is not None
            and (image_file := self._filename_in_folder(self.test_image_folder, image_name, image=True))
        ):
            label_file = self.test_label_folder / (image_name + ".txt")
        else:
            log.fatal(f"Cannot find image {image_name} in train/validation/test dataset!")
            exit()

        if not label_file.is_file():
            log.debug(f"Could not find annotation for {image_name}, assuming empty")
            return []

        with ExifToolHelper() as et:
            img_metadata = et.get_metadata(image_file, ("-ImageWidth", "-ImageHeight"))[0]

        if "File:ImageWidth" in img_metadata and "File:ImageHeight" in img_metadata:
            img_size = np.array([img_metadata["File:ImageWidth"], img_metadata["File:ImageHeight"]], dtype=np.uint16)
        else:
            img_size = np.array([img_metadata["EXIF:ImageWidth"], img_metadata["EXIF:ImageHeight"]], dtype=np.uint16)

        label_file_content = label_file.read_text().splitlines()
        labels = []

        for label in label_file_content:
            label_elements = label.split(" ")

            class_name = self.class_names[int(label_elements[0])]
            x, y, w, h = map(float, label_elements[1:5])

            labels.append(
                Annotation(
                    class_name,
                    x * img_size[0] - w * img_size[0] // 2,
                    y * img_size[1] - h * img_size[1] // 2,
                    w * img_size[0],
                    h * img_size[1],
                    class_name,
                    image_name,
                    img_size,
                    [],
                )
            )

        return labels

    def write_dataset_file(self) -> None:
        data_content = {
            "train": str(self.train_image_folder.relative_to(self.dataset_root_folder)),
            "val": str(self.validation_image_folder.relative_to(self.dataset_root_folder)),
            "test": str(self.test_image_folder.relative_to(self.dataset_root_folder)) if self.test_image_folder else None,
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        with (self.dataset_root_folder / "data.yml").open("w") as data_file_writer:
            safe_dump(data_content, data_file_writer)

    def write_classes_file(self) -> None:
        (self.train_label_folder / "classes.txt").write_text("\n".join(self.class_names))
        (self.validation_label_folder / "classes.txt").write_text("\n".join(self.class_names))
        if self.test_label_folder is not None:
            (self.test_label_folder / "classes.txt").write_text("\n".join(self.class_names))

    def write_label_file(self, image_name: str, labels: list[Annotation]) -> None:
        # Find image name to define the label folder for this label
        if self._filename_in_folder(self.train_image_folder, image_name):
            label_file = self.train_label_folder / (image_name + ".txt")
        elif self._filename_in_folder(self.validation_image_folder, image_name):
            label_file = self.validation_label_folder / (image_name + ".txt")
        elif (
            self.test_image_folder is not None
            and self.test_label_folder is not None
            and self._filename_in_folder(self.test_image_folder, image_name)
        ):
            label_file = self.test_label_folder / (image_name + ".txt")
        else:
            raise RuntimeError(f"Cannot write label file because {image_name} is not a train/validation/test image!")

        write_yolo_annotation_file(label_file, labels, self.class_names)

    @classmethod
    def from_dataset_file(cls, dataset_file: Path) -> YoloDataset:
        with dataset_file.open("r") as dataset_file_handler:
            data: dict[str, Any] = safe_load(dataset_file_handler)

        def get_image_path(subset: str) -> str:
            return data.get(subset)  # type: ignore[return-value]

        return cls(
            dataset_file.parent,
            dataset_file.parent / get_image_path("train"),
            dataset_file.parent / get_image_path("val"),
            dataset_file.parent / get_image_path("test") if data.get("test") else None,
            dataset_file.parent / get_image_path("train").replace("images", "labels"),
            dataset_file.parent / get_image_path("val").replace("images", "labels"),
            dataset_file.parent / get_image_path("test").replace("images", "labels") if data.get("test") else None,
            data["names"],
        )

    @classmethod
    def from_root(cls, root_path: Path, class_names: list[str], use_test: bool = False) -> YoloDataset:
        return cls(
            root_path,
            root_path / "images" / "train",
            root_path / "images" / "val",
            root_path / "images" / "test" if use_test else None,
            root_path / "labels" / "train",
            root_path / "labels" / "val",
            root_path / "labels" / "test" if use_test else None,
            class_names,
        )

    @staticmethod
    def _filename_in_folder(folder: Path, filename: str, image: bool = False) -> Path | None:
        for file in folder.iterdir():
            if file.is_file() and file.stem == filename:
                if image and file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                return file
        return None


def load_yolo_detections(image_folder: Path | list[Path], detections_folder: Path, class_names: list[str]) -> dict[str, list[Detection]]:
    yolo_data: dict[str, list[Detection]] = {}

    imgs = get_image_paths(image_folder)

    # Do this at once, saves a lot of time
    with ExifToolHelper() as et:
        imgs_metadata = et.get_metadata(imgs, ("-ImageWidth", "-ImageHeight"))

    for img_file, img_metadata in zip(imgs, imgs_metadata):
        label_file = detections_folder / (img_file.stem + ".txt")

        if "File:ImageWidth" in img_metadata and "File:ImageHeight" in img_metadata:
            img_size = np.array([img_metadata["File:ImageWidth"], img_metadata["File:ImageHeight"]], dtype=np.uint16)
        else:
            img_size = np.array([img_metadata["EXIF:ImageWidth"], img_metadata["EXIF:ImageHeight"]], dtype=np.uint16)

        if not label_file.is_file():
            log.debug(f"Could not find annotation for {img_file.name}, assuming empty")
            yolo_data[img_file.stem] = []
            continue

        label_file_content = label_file.read_text().splitlines()
        detections = []

        for label in label_file_content:
            label_elements = label.split(" ")

            class_name = class_names[int(label_elements[0])]
            x, y, w, h = map(float, label_elements[1:5])
            confidence = float(label_elements[5])

            detections.append(
                Detection(
                    x * img_size[0] - w * img_size[0] // 2,
                    y * img_size[1] - h * img_size[1] // 2,
                    w * img_size[0],
                    h * img_size[1],
                    class_name,
                    confidence,
                    img_file.stem,
                    img_size,
                )
            )

        yolo_data[img_file.stem] = detections

    return yolo_data


def load_yolo_annotations(
    dataset_file_or_image_folder: Path, label_folder: Path | None = None, class_names: list[str] | None = None
) -> tuple[dict[str, list[Annotation]], list[str]]:
    if label_folder is None and class_names is None:
        return _load_yolo_annotations_from_dataset_file(dataset_file_or_image_folder)
    elif class_names is None:
        assert label_folder is not None
        return _load_yolo_annotations_from_folder(dataset_file_or_image_folder, label_folder)

    assert label_folder is not None
    return _load_yolo_annotations_with_class_names(dataset_file_or_image_folder, label_folder, class_names)


def _load_yolo_annotations_from_dataset_file(dataset_file: Path) -> tuple[dict[str, list[Annotation]], list[str]]:
    yolo_dataset = YoloDataset.from_dataset_file(dataset_file)
    yolo_data: dict[str, list[Annotation]] = {}

    for k in yolo_dataset.image_paths.keys():
        image_path = yolo_dataset.image_paths[k]
        label_path = yolo_dataset.label_paths[k]

        if image_path is not None and label_path is not None:
            yolo_data.update(load_yolo_annotations(image_path, label_path, yolo_dataset.class_names)[0])

    return yolo_data, yolo_dataset.class_names


def _load_yolo_annotations_from_folder(image_folder: Path, label_folder: Path) -> tuple[dict[str, list[Annotation]], list[str]]:
    classes_file = label_folder / "classes.txt"

    if classes_file.is_file():
        class_names = classes_file.read_text().splitlines()
    else:
        log.warning("Class names file does not exist! Generating numbers instead of names")
        class_names = [f"class_{i}" for i in range(100)]

    return load_yolo_annotations(image_folder, label_folder, class_names)


def _load_yolo_annotations_with_class_names(
    image_folder: Path, label_folder: Path, class_names: list[str]
) -> tuple[dict[str, list[Annotation]], list[str]]:
    yolo_data: dict[str, list[Annotation]] = {}

    imgs = get_image_paths(image_folder)

    # Do this at once, saves a lot of time
    with ExifToolHelper() as et:
        imgs_metadata = et.get_metadata(imgs, ("-ImageWidth", "-ImageHeight"))

    class_counter: dict[str, int] = defaultdict(int)

    for img_file, img_metadata in zip(imgs, imgs_metadata):
        label_file = label_folder / (img_file.stem + ".txt")

        if "File:ImageWidth" in img_metadata and "File:ImageHeight" in img_metadata:
            img_size = np.array([img_metadata["File:ImageWidth"], img_metadata["File:ImageHeight"]], dtype=np.uint16)
        else:
            img_size = np.array([img_metadata["EXIF:ImageWidth"], img_metadata["EXIF:ImageHeight"]], dtype=np.uint16)

        if not label_file.is_file():
            log.debug(f"Could not find annotation for {img_file.name}, assuming empty")
            yolo_data[img_file.stem] = []
            continue

        label_file_content = label_file.read_text().splitlines()
        annotations = []

        for label in label_file_content:
            label_elements = label.split(" ")

            class_name = class_names[int(label_elements[0])]
            name = f"{class_name}_{class_counter[class_name]}"
            x, y, w, h = map(float, label_elements[1:5])

            # If available, add detection confidence
            if len(label_elements) > 5:
                name += f"_{label_elements[5]}"

            annotations.append(
                Annotation(
                    name,
                    x * img_size[0] - w * img_size[0] // 2,
                    y * img_size[1] - h * img_size[1] // 2,
                    w * img_size[0],
                    h * img_size[1],
                    class_name,
                    img_file.stem,
                    img_size,
                    [],
                )
            )

            class_counter[class_name] += 1

        yolo_data[img_file.stem] = annotations

    return yolo_data, class_names


def write_yolo_annotations(output_folder: Path, annotation_dict: dict[str, list[Annotation]], class_names: list[str]) -> None:
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)

    for file_name, annotations in annotation_dict.items():
        label_file = output_folder / (file_name + ".txt")
        write_yolo_annotation_file(label_file, annotations, class_names)


def write_yolo_annotation_file(output_file: Path, annotations: list[Annotation], class_names: list[str]) -> None:
    lines = []
    for annotation in annotations:
        lines.append(f"{class_names.index(annotation.class_name)} " + " ".join(map(str, list(annotation.yolo_xywh))))
    output_file.write_text("\n".join(lines))


def write_yolo_dataset_file(
    output_file: Path,
    class_names: list[str],
    train_folder_name: str = "train",
    val_folder_name: str = "val",
    test_folder_name: str | None = "test",
) -> None:
    data_content = {
        "train": f"images/{train_folder_name}",
        "val": f"images/{val_folder_name}",
        "test": f"images/{test_folder_name}" if test_folder_name else None,
        "nc": len(class_names),
        "names": list(class_names),
    }

    with output_file.open("w") as data_file_writer:
        safe_dump(data_content, data_file_writer)
