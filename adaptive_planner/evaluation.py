from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from ultralytics.utils.metrics import bbox_ioa

from adaptive_planner import setup_logging
from adaptive_planner.predict import Detection
from adaptive_planner.utils import Annotation, create_distance_matrix, get_location_by_name

if TYPE_CHECKING:
    from typing import SupportsIndex

    from numpy.typing import NDArray

    from adaptive_planner.location import Location


log = setup_logging(__name__)


@dataclass
class EvaluationResult:
    y_true: list[str]
    y_pred: list[str]
    conf: list[float | None]
    names: list[tuple[str | None, str | None]]

    labels: list[str] = field(default_factory=lambda: ["F", "W", "background"])

    def __or__(self, other: EvaluationResult) -> EvaluationResult:
        return EvaluationResult(self.y_true + other.y_true, self.y_pred + other.y_pred, self.conf + other.conf, self.names + other.names)

    @property
    def precision(self) -> float:
        precision_dict = {}
        for k, v in self.get_tp_fp_fn_indices().items():
            precision_dict[k] = len(v[0]) / (len(v[0]) + len(v[1]) + 1e-16)

        return np.mean(list(precision_dict.values())).item()

    @property
    def recall(self) -> float:
        recall_dict = {}
        for k, v in self.get_tp_fp_fn_indices().items():
            recall_dict[k] = len(v[0]) / (len(v[0]) + len(v[2]) + 1e-16)

        return np.mean(list(recall_dict.values())).item()

    @property
    def f1(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall + 1e-16)

    @property
    def confusion_matrix(self) -> NDArray[np.float32]:
        return confusion_matrix(self.y_true, self.y_pred, labels=self.labels)  # type: ignore[no-any-return]

    def pop(self, i: SupportsIndex) -> None:
        self.y_true.pop(i)
        self.y_pred.pop(i)
        self.conf.pop(i)
        self.names.pop(i)

    def make_class_agnostic(self, single_class_name: str = "object") -> EvaluationResult:
        result = deepcopy(self)
        result.y_true = ["background" if class_name == "background" else single_class_name for class_name in result.y_true]
        result.y_pred = ["background" if class_name == "background" else single_class_name for class_name in result.y_pred]
        result.labels = [single_class_name, "background"]
        return result

    def get_tp_fp_fn_indices(self) -> dict[str, tuple[list[int], list[int], list[int]]]:
        _y_true = np.array(self.y_true, dtype=str)
        _y_pred = np.array(self.y_pred, dtype=str)

        tp_fp_fn = {}
        for class_name in self.labels:
            if class_name == "background":
                continue

            tp_indices = np.where((_y_pred == class_name) & (_y_true == class_name))[0].tolist()
            fp_indices = np.where((_y_pred == class_name) & (_y_true != class_name))[0].tolist()
            fn_indices = np.where((_y_pred != class_name) & (_y_true == class_name))[0].tolist()

            tp_fp_fn[class_name] = (tp_indices, fp_indices, fn_indices)
        return tp_fp_fn

    @classmethod
    def from_georeferenced_image_detections(
        cls: type[EvaluationResult],
        gt_locations: list[Location],
        detected_locations: list[Location],
        dist_threshold: float = 0.3,
        conf_threshold: float = 0.5,
        border_region: float = 0.3,
        use_uncertainty_measure_as_confidence: bool = False,
    ) -> EvaluationResult:
        gt_locations_per_image: dict[str, list[Location]] = defaultdict(list)
        dt_locations_per_image: dict[str, list[Location]] = defaultdict(list)

        for gt_location in gt_locations:
            gt_location = gt_location.copy()
            gt_location.properties["_image_name"] = gt_location.properties.pop("image_name")
            gt_locations_per_image[gt_location.properties["_image_name"]].append(gt_location)

        for dt_location in detected_locations:
            dt_location = dt_location.copy()
            dt_location.properties["_image_name"] = dt_location.properties.pop("image_name")
            dt_location.properties["best_class_name"] = dt_location.properties.pop("class_name")
            dt_location.properties["best_confidence"] = (
                dt_location.properties.get("uncertainty_measure", dt_location.properties["confidence"])
                if use_uncertainty_measure_as_confidence
                else dt_location.properties["confidence"]
            )
            dt_location.properties["status"] = "INVESTIGATED"
            dt_locations_per_image[dt_location.properties["_image_name"]].append(dt_location)

        evaluation_result = EvaluationResult([], [], [], [])
        for image_name in set(gt_locations_per_image.keys()).union(dt_locations_per_image.keys()):
            _eval_result = EvaluationResult.from_georeferenced_detections(
                gt_locations_per_image[image_name],
                dt_locations_per_image[image_name],
                dist_threshold=dist_threshold,
                conf_threshold=conf_threshold,
            )

            # Remove FP and FN when they are within the border region (in meters). They are probably caused by inaccuracies in the
            # GPS geo-referencing or by the manual measurements
            _tp_fp_fn_indices = _eval_result.get_tp_fp_fn_indices()
            fp = [item for fp_indices in _tp_fp_fn_indices.values() for item in fp_indices[1]]
            fn = [item for fn_indices in _tp_fp_fn_indices.values() for item in fn_indices[2]]

            indices_to_remove = []

            # Filter out FPs that are close to the border of the image. Keep FPs that are caused by mismatching labels, since they
            # are visible in the image.
            for fp_i in fp:
                detection_name = _eval_result.names[fp_i][0]
                assert detection_name is not None

                gt_name = _eval_result.names[fp_i][1]
                if gt_name is not None:
                    continue

                detection_location = get_location_by_name(dt_locations_per_image[image_name], detection_name)
                if detection_location.properties["distance_to_border"] <= border_region:
                    indices_to_remove.append(fp_i)

            # Filter out FNs that are close to the border of the image. Keep FNs that are caused by mismatching labels, since they
            # are visible in the image.
            for fn_i in fn:
                detection_name = _eval_result.names[fn_i][0]
                if detection_name is not None:
                    continue

                gt_name = _eval_result.names[fn_i][1]
                assert gt_name is not None

                gt_location = get_location_by_name(gt_locations_per_image[image_name], gt_name)
                if gt_location.properties["distance_to_border"] <= border_region:
                    indices_to_remove.append(fn_i)

            for i in sorted(indices_to_remove, reverse=True):
                _eval_result.pop(i)

            evaluation_result = evaluation_result | _eval_result

        return evaluation_result

    @classmethod
    def from_georeferenced_detections(
        cls: type[EvaluationResult],
        gt_locations: list[Location],
        detected_locations: list[Location],
        dist_threshold: float = 0.3,
        conf_threshold: float = 0.5,
    ) -> EvaluationResult:
        if (len(gt_locations) > 0 and "image_name" in gt_locations[0].properties) or (
            len(detected_locations) > 0 and "image_name" in detected_locations[0].properties
        ):
            raise RuntimeError("Run from_georeferenced_image_detections() instead!")

        detected_locations = [dl for dl in detected_locations if dl.properties["best_confidence"] >= conf_threshold]
        detected_locations = [dl for dl in detected_locations if dl.properties["status"] != "REJECTED"]

        distance_matrix = create_distance_matrix(detected_locations, gt_locations, dist_threshold)
        dt_i, gt_j = linear_sum_assignment(distance_matrix)

        y_true = []
        y_pred = []
        conf = []
        names = []
        for i, j in zip(dt_i, gt_j):
            # TP
            if float(distance_matrix[i, j]) <= dist_threshold:
                y_true.append(gt_locations[j].properties["class_name"])
                y_pred.append(detected_locations[i].properties["best_class_name"])
                conf.append(detected_locations[i].properties["best_confidence"])
                names.append((detected_locations[i].properties["name"], gt_locations[j].properties["name"]))

            # FP and FN
            else:
                y_true.append("background")
                y_pred.append(detected_locations[i].properties["best_class_name"])
                conf.append(detected_locations[i].properties["best_confidence"])
                names.append((detected_locations[i].properties["name"], None))

                y_true.append(gt_locations[j].properties["class_name"])
                y_pred.append("background")
                conf.append(None)
                names.append((None, gt_locations[j].properties["name"]))

        # FP
        for i in range(len(detected_locations)):
            if i not in dt_i:
                y_true.append("background")
                y_pred.append(detected_locations[i].properties["best_class_name"])
                conf.append(detected_locations[i].properties["best_confidence"])
                names.append((detected_locations[i].properties["name"], None))

        # FN
        for j in range(len(gt_locations)):
            if j not in gt_j:
                y_true.append(gt_locations[j].properties["class_name"])
                y_pred.append("background")
                conf.append(None)
                names.append((None, gt_locations[j].properties["name"]))

        return cls(y_true, y_pred, conf, names)

    @classmethod
    def from_detections_with_annotations(
        cls: type[EvaluationResult],
        detection_dict: dict[str, list[Detection]],
        annotation_dict: dict[str, list[Annotation]],
        min_iou: float = 0.2,
    ) -> EvaluationResult:
        image_names = list(set(detection_dict.keys()) | set(annotation_dict.keys()))

        y_true: list[str] = []
        y_pred: list[str] = []
        conf: list[float | None] = []
        names: list[tuple[str | None, str | None]] = []

        for image_name in image_names:
            detections = detection_dict[image_name]
            annotations = annotation_dict[image_name]

            iou_matrix = np.empty((len(detections), len(annotations)), dtype=np.float32)
            for i, detection in enumerate(detections):
                for j, annotation in enumerate(annotations):
                    # TODO: check if this can be done using all detections and annotations and create
                    # the distance matrix directly
                    iou = bbox_ioa(
                        np.expand_dims(np.array(detection.x1y1x2y2, dtype=np.float32), axis=0),
                        np.expand_dims(np.array(annotation.x1y1x2y2, dtype=np.float32), axis=0),
                        iou=True,
                    )[0][0]
                    iou_matrix[i, j] = iou if iou > min_iou else -1e7

            detections_i, annotations_j = linear_sum_assignment(iou_matrix, maximize=True)
            for i, j in zip(detections_i, annotations_j):
                # TP
                if float(iou_matrix[i, j]) >= min_iou:
                    y_true.append(annotations[j].class_name)
                    y_pred.append(detections[i].class_name)
                    conf.append(detections[i].confidence)

                # FP and FN
                else:
                    y_true.append(annotations[j].class_name)
                    y_pred.append("background")
                    conf.append(None)

                    y_true.append("background")
                    y_pred.append(detections[i].class_name)
                    conf.append(detections[i].confidence)

            # FP
            for i in range(len(detections)):
                if i not in detections_i:
                    y_true.append("background")
                    y_pred.append(detections[i].class_name)
                    conf.append(detections[i].confidence)

            # FN
            for j in range(len(annotations)):
                if j not in annotations_j:
                    y_true.append(annotations[j].class_name)
                    y_pred.append("background")
                    conf.append(None)

        return cls(
            y_true,
            y_pred,
            conf,
            names,
        )


def pixel_within_border_region(coordinate: NDArray[np.uint16], image_size: NDArray[np.uint16], border_size: int) -> bool:
    return bool(np.all((coordinate >= border_size) & (coordinate <= (image_size - border_size))))
