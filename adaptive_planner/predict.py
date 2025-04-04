from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import torch as th
import torch.nn.functional as F
from torchvision.ops import nms

from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.ops import convert_torch2numpy_batch, nms_rotated, scale_boxes, xywh2xyxy

from adaptive_planner import setup_logging
from adaptive_planner.uncertainty_estimation import (
    UncertaintyEstimatorCLS,
    UncertaintyEstimatorCLS2,
    UncertaintyEstimatorIOU,
    UncertaintyEstimatorOCC,
    UncertaintyEstimatorYOLO,
    UncertaintyEstimatorYOLOMean,
    UncertaintyProduct,
)

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray
    from torch.nn import Module


log = setup_logging(__name__)


@dataclass
class Detection:
    x: int  # Top-left
    y: int  # Top-left
    w: int
    h: int
    class_name: str
    confidence: float
    image_name: str
    image_size: NDArray[np.uint16]
    class_confidences: dict[str, float] | None = None
    uncertainty_measure: float | None = None

    @property
    def coordinate(self) -> NDArray[np.uint16]:
        return np.array([self.x + self.w // 2, self.y + self.h // 2], dtype=np.uint16)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (
            *self.coordinate,
            self.w,
            self.h,
        )

    @property
    def yolo_xywh(self) -> tuple[float, float, float, float]:
        return (
            self.coordinate[0] / self.image_size[0],
            self.coordinate[1] / self.image_size[1],
            self.w / self.image_size[0],
            self.h / self.image_size[1],
        )

    @property
    def x1y1x2y2(self) -> tuple[float, float, float, float]:
        return (
            self.x,
            self.y,
            self.x + self.w,
            self.y + self.h,
        )


class CustomResults(Results):  # type: ignore[misc]
    def __init__(
        self,
        orig_img: NDArray[np.uint8],
        path: str,
        names: dict[int, str],
        boxes: th.Tensor | None = None,
        masks: th.Tensor | None = None,
        probs: th.Tensor | None = None,
        keypoints: th.Tensor | None = None,
        obb: th.Tensor | None = None,
        speed: dict[str, float] | None = None,
        class_confidences: th.Tensor | None = None,
        uncertainty_measure: th.Tensor | None = None,
    ) -> None:
        super().__init__(orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)

        self.class_confidences = class_confidences
        self.uncertainty_measure = uncertainty_measure

        self._keys = ["boxes", "class_confidences", "uncertainty_measure"]

    def new(self) -> CustomResults:
        return CustomResults(self.orig_img, self.path, self.names, speed=self.speed, uncertainty_measure=self.uncertainty_measure)

    def to_detections(self) -> list[Detection]:
        assert self.boxes is not None, "Can only convert detection task to Detections!"
        boxes = self.boxes.cpu()

        detections = []
        for i in range(len(self.boxes)):
            x, y, w, h = boxes.xywh.numpy()[i, :].round().astype(int)
            class_confidences = (
                None
                if self.class_confidences is None
                else {self.names[j]: self.class_confidences[i, j].item() for j in range(len(self.names))}
            )
            detection = Detection(
                x,
                y,
                w,
                h,
                self.names[boxes.cls[i].item()],
                boxes.conf[i].item(),
                Path(self.path).stem,
                np.flip(self.orig_shape).astype(np.uint16),
                class_confidences=class_confidences,
                uncertainty_measure=None if self.uncertainty_measure is None else self.uncertainty_measure[i].item(),
            )

            detections.append(detection)

        return detections


class CustomPredictor(BasePredictor):  # type: ignore[misc]
    def __init__(self, cfg: str = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: Any = None) -> None:
        super().__init__(cfg, overrides, _callbacks)

    def postprocess(
        self, preds: th.Tensor, img: th.Tensor, orig_imgs: th.Tensor | NDArray[np.uint8] | list[NDArray[np.uint8]]
    ) -> list[CustomResults]:
        """Post-processes predictions and returns a list of Results objects."""
        nms_preds, cls_conf = self.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, (list, np.ndarray)):  # input images are a torch.Tensor, not a list
            orig_imgs = convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(nms_preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs[i, :]
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                CustomResults(
                    orig_img,  # type:ignore[arg-type]
                    img_path,
                    self.model.names,
                    boxes=pred,
                    class_confidences=cls_conf[i],
                )
            )

        return results

    @staticmethod
    def non_max_suppression(
        prediction: list[th.Tensor] | th.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: list[int] | None = None,
        agnostic: bool = False,
        multi_label: bool = False,
        labels: list[list[int | th.Tensor]] = [],
        max_det: int = 300,
        nc: int | None = 0,
        max_time_img: float = 0.05,
        max_nms: int = 30000,
        max_wh: int = 7680,
        in_place: bool = True,
        rotated: bool = False,
    ) -> tuple[list[th.Tensor], list[th.Tensor]]:
        # Checks
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output
        if classes is not None:
            classes = th.tensor(classes, device=prediction.device)  # type: ignore[assignment]

        if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
            output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
            if classes is not None:
                output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
            return output, []

        bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4  # number of masks
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        if not rotated:
            if in_place:
                prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
            else:
                prediction = th.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

        t = time()
        output = [th.zeros((0, 6 + nm), device=prediction.device)] * bs
        output_cls = [th.zeros((0, nc), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]) and not rotated:
                lb = labels[xi]
                v = th.zeros((len(lb), nc + nm + 4), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])  # type: ignore[call-overload]
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # type: ignore[call-overload]
                x = th.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box, cls, mask = x.split((4, nc, nm), 1)

            if multi_label:
                i, j = th.where(cls > conf_thres)
                x = th.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = th.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            assert x.size(0) == cls.size(0)

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == classes).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                sorted_idx = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
                x = x[sorted_idx]
                cls = cls[sorted_idx]

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            if rotated:
                boxes = th.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
                i = nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c  # boxes (offset by class)
                i = nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            output[xi] = x[i]
            output_cls[xi] = th.softmax(cls[i], dim=1)
            if (time() - t) > time_limit:
                LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
                break  # time limit exceeded

        return output, output_cls


class MCD_STRATEGY(Enum):
    EARLY = 0
    LATE = 1


class UNCERTAINTY_METHOD(Enum):
    OCC = 0
    CLS = 1
    CLS2 = 2
    IOU = 3
    YOLO = 4
    YOLO_MEAN = 5
    OCC_CLS = 6
    OCC_IOU = 7
    CLS_IOU = 8
    OCC_CLS_IOU = 9
    OCC_CLS2_IOU = 10


class MonteCarloDropoutUncertaintyPredictor(CustomPredictor):
    def __init__(
        self,
        cfg: str = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: Any = None,
        dropout_iterations: int = 20,
        dropout_probability: float = 0.2,
        dropout_strategy: MCD_STRATEGY = MCD_STRATEGY.EARLY,
        uncertainty_method: UNCERTAINTY_METHOD = UNCERTAINTY_METHOD.OCC,
    ) -> None:
        super().__init__(cfg, overrides, _callbacks)

        self.inference_iterations = dropout_iterations + 1  # One additional one for the inference without dropout
        self.dropout_probability = dropout_probability

        self.dropout_strategy = dropout_strategy

        self.uncertainty_estimator: Module
        match uncertainty_method:
            case UNCERTAINTY_METHOD.OCC:
                self.uncertainty_estimator = UncertaintyEstimatorOCC()
            case UNCERTAINTY_METHOD.CLS:
                self.uncertainty_estimator = UncertaintyEstimatorCLS()
            case UNCERTAINTY_METHOD.CLS2:
                self.uncertainty_estimator = UncertaintyEstimatorCLS2()
            case UNCERTAINTY_METHOD.IOU:
                self.uncertainty_estimator = UncertaintyEstimatorIOU()
            case UNCERTAINTY_METHOD.YOLO:
                self.uncertainty_estimator = UncertaintyEstimatorYOLO()
            case UNCERTAINTY_METHOD.YOLO_MEAN:
                self.uncertainty_estimator = UncertaintyEstimatorYOLOMean()
            case UNCERTAINTY_METHOD.OCC_CLS:
                self.uncertainty_estimator = UncertaintyProduct([UncertaintyEstimatorOCC, UncertaintyEstimatorCLS])
            case UNCERTAINTY_METHOD.OCC_IOU:
                self.uncertainty_estimator = UncertaintyProduct([UncertaintyEstimatorOCC, UncertaintyEstimatorIOU])
            case UNCERTAINTY_METHOD.CLS_IOU:
                self.uncertainty_estimator = UncertaintyProduct([UncertaintyEstimatorCLS, UncertaintyEstimatorIOU])
            case UNCERTAINTY_METHOD.OCC_CLS_IOU:
                self.uncertainty_estimator = UncertaintyProduct([UncertaintyEstimatorOCC, UncertaintyEstimatorCLS, UncertaintyEstimatorIOU])
            case UNCERTAINTY_METHOD.OCC_CLS2_IOU:
                self.uncertainty_estimator = UncertaintyProduct(
                    [UncertaintyEstimatorOCC, UncertaintyEstimatorCLS2, UncertaintyEstimatorIOU]
                )
            case _:
                raise NotImplementedError

        print(self)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}(dropout_strategy={self.dropout_strategy.name},"
            f"dropout_probability={self.dropout_probability},uncertainty_estimator={self.uncertainty_estimator},"
            f"inference_iterations={self.inference_iterations})"
        )

    def postprocess(
        self, preds: th.Tensor, img: th.Tensor, orig_imgs: th.Tensor | NDArray[np.uint8] | list[NDArray[np.uint8]]
    ) -> list[CustomResults]:
        nms_preds, cls_conf = self.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        assert len(nms_preds) % self.inference_iterations == 0

        if not isinstance(orig_imgs, (list, np.ndarray)):  # input images are a torch.Tensor, not a list
            orig_imgs = convert_torch2numpy_batch(orig_imgs)

        n_batch = len(nms_preds) // self.inference_iterations

        results = []
        for i in range(n_batch):
            pred = nms_preds[i]
            mcd_pred = [nms_preds[i + j * n_batch] for j in range(1, self.inference_iterations)]
            mcd_conf = [cls_conf[i + j * n_batch] for j in range(1, self.inference_iterations)]

            uncertainty = self.uncertainty_estimator(pred, mcd_pred, mcd_conf)

            orig_img = (
                orig_imgs[i // self.inference_iterations] if isinstance(orig_imgs, list) else orig_imgs[i // self.inference_iterations, :]
            )
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(
                CustomResults(
                    orig_img,  # type:ignore[arg-type]
                    img_path,
                    self.model.names,
                    boxes=pred,
                    class_confidences=cls_conf[i],
                    uncertainty_measure=uncertainty,
                )
            )

        return results

    def inference(self, im: th.Tensor, *args: list[Any], **kwargs: Any) -> th.Tensor:
        y: list[th.Tensor | None] = [None] * len(self.model.model.model)

        match self.dropout_strategy:
            # Apply dropout at skip connections from layers 4 and 6 and before SPPF
            case MCD_STRATEGY.EARLY:
                fixed_layer = 8
                dropout_layers = {
                    9: [-1],
                    11: [6],
                    14: [4],
                }

            # Apply dropout before the DETECT head
            case MCD_STRATEGY.LATE:
                fixed_layer = 21
                dropout_layers = {
                    22: [15, 18, 21],
                }

            case _:
                raise NotImplementedError

        # Run backbone
        x0 = im
        for m in self.model.model.model[: fixed_layer + 1]:
            if m.f != -1:  # if not from previous layer
                # Get data from from earlier layers
                x0 = y[m.f] if isinstance(m.f, int) else [x0 if j == -1 else y[j] for j in m.f]  # type: ignore[assignment]

            x0 = m(x0)

            if m.i in self.model.model.save:
                y[m.i] = x0  # save output

        output = []
        for i in range(self.inference_iterations):
            x = x0.clone().detach()

            for m in self.model.model.model[fixed_layer + 1 :]:
                if m.f != -1:  # if not from previous layer
                    # Get data from earlier layers
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # type: ignore[assignment]

                if i > 0 and m.i in dropout_layers.keys():
                    if isinstance(m.f, list):
                        for dc in dropout_layers[m.i]:
                            x[m.f.index(dc)] = F.dropout(x[m.f.index(dc)].clone().detach(), p=self.dropout_probability, training=True)
                    else:
                        assert len(dropout_layers[m.i]) == 1 and dropout_layers[m.i][0] == -1
                        x = F.dropout(x, p=self.dropout_probability, training=True)

                x = m(x)

                if m.i in self.model.model.save:
                    y[m.i] = x

            output.append(x[0])

        return th.cat(output, dim=0)  # Concat as batch


if __name__ == "__main__":
    from functools import partial

    from ultralytics import YOLO

    th.set_printoptions(sci_mode=False)  # type: ignore[no-untyped-call]

    images = [
        # "/home/abe/data/20240213_clustered_1/DJI_202402131154_010_MappingCluster12m/DJI_20240213122010_0007.JPG",
        "/home/abe/data/20240213_clustered_1/DJI_202402131154_008_MappingCluster12m/DJI_20240213115805_0011.JPG",
        # "/home/abe/data/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122013_0008.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122015_0009.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122018_0010.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122021_0011.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122024_0012.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122026_0013.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122029_0014.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122032_0015.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122035_0016.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122037_0017.JPG",
        # "/media/rick/DATA_RICK/adaptive-planning/20240213_clustered_1/DJI_202402131154_010_MappingCluster32m/DJI_20240213122040_0018.JPG"
    ]

    model = YOLO("adaptive_planner/best_n.pt")

    results: list[CustomResults] = model.predict(
        images,
        predictor=partial(
            MonteCarloDropoutUncertaintyPredictor,
            dropout_iterations=20,
            dropout_probability=0.75,
            dropout_strategy=MCD_STRATEGY.LATE,
            uncertainty_method=UNCERTAINTY_METHOD.CLS2,
        ),
        imgsz=2048,
        conf=0.1,
    )

    result = results[0].cpu()
    print(result.boxes)
    print(result.class_confidences)
    print(result.uncertainty_measure)
