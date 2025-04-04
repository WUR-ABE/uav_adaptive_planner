from __future__ import annotations

from functools import partial
from json import dump
from pathlib import Path
from tap import Tap
from typing import TYPE_CHECKING
from warnings import simplefilter

from codecarbon import EmissionsTracker
from shapely import Polygon
from tqdm.auto import tqdm
from ultralytics import YOLO

from adaptive_planner import logging_redirect_tqdm, setup_logging
from adaptive_planner.io import get_image_paths
from adaptive_planner.io.kml import write_gps_locations_to_kml
from adaptive_planner.predict import MCD_STRATEGY, UNCERTAINTY_METHOD, CustomPredictor, MonteCarloDropoutUncertaintyPredictor

from .metashape import get_camera_by_name, load_project, pixel_to_gps_location

if TYPE_CHECKING:
    from typing import Any

    from adaptive_planner.predict import CustomResults


log = setup_logging(__name__)
simplefilter(action="ignore", category=FutureWarning)


class ArgumentParser(Tap):
    weights_file: Path  # Path to the YOLO network weigth
    agisoft_project_file: Path  # Path to the Agisoft Metashape project (*.psx) file
    output_file: Path  # Output kml file
    image_folders: list[Path]  # Path image folder to evaluate
    uncertainty_method: str = "yolo_uncertainty"  # Uncertainty method
    chunk_name: str | None = None  # Name of the chunk in Agisoft
    imgsz: int = 2048  # Image size for detection
    nms_iou: float = 0.7  # Non-maximum suppression IoU threshold
    batch_size: int = 12  # Batch size for detection
    dropout_probability: float = 0.75  # Dropout probability for MCD

    def configure(self) -> None:
        self.add_argument("weights_file")
        self.add_argument("agisoft_project_file")
        self.add_argument("output_file")
        self.add_argument("image_folders", nargs="+")

    def process_args(self) -> None:
        if not self.weights_file.is_file():
            raise FileNotFoundError(f"Weights file {self.weights_file.name} does not exist!")

        if not self.agisoft_project_file.is_file():
            raise FileNotFoundError(f"Agisoft project file {self.agisoft_project_file.name} does not exist!")

        if not all([f.is_dir() for f in self.image_folders]):
            raise FileNotFoundError("Not all image folders exist!")


def main() -> None:
    args = ArgumentParser().parse_args()

    network = YOLO(args.weights_file)
    chunk = load_project(args.agisoft_project_file, chunk_name=args.chunk_name)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    if chunk is None:
        raise RuntimeError("Could not load Agisoft project")

    # Get predictor
    if args.uncertainty_method.startswith("yolo_"):
        predictor = CustomPredictor
    elif args.uncertainty_method.startswith("mcd_"):
        predictor = MonteCarloDropoutUncertaintyPredictor
    else:
        raise NotImplementedError

    # Get predictor arguments
    predictor_kwargs: dict[str, Any] = {}
    if args.uncertainty_method.startswith("mcd_"):
        predictor_kwargs["dropout_probability"] = args.dropout_probability

        if args.uncertainty_method.startswith("mcd_early_"):
            predictor_kwargs["dropout_strategy"] = MCD_STRATEGY.EARLY
        elif args.uncertainty_method.startswith("mcd_late_"):
            predictor_kwargs["dropout_strategy"] = MCD_STRATEGY.LATE
        else:
            raise NotImplementedError

        UNCERTAINTY_METHOD_MAP = {
            "occ_cls2_iou": UNCERTAINTY_METHOD.OCC_CLS2_IOU,
            "occ_cls_iou": UNCERTAINTY_METHOD.OCC_CLS_IOU,
            "cls_iou": UNCERTAINTY_METHOD.CLS_IOU,
            "occ_cls": UNCERTAINTY_METHOD.OCC_CLS,
            "occ_iou": UNCERTAINTY_METHOD.OCC_IOU,
            "occ": UNCERTAINTY_METHOD.OCC,
            "cls2": UNCERTAINTY_METHOD.CLS2,
            "cls": UNCERTAINTY_METHOD.CLS,
            "iou": UNCERTAINTY_METHOD.IOU,
            "yolo_mean": UNCERTAINTY_METHOD.YOLO_MEAN,
        }

        # Check each mapping and set the appropriate method
        for suffix, method in UNCERTAINTY_METHOD_MAP.items():
            if args.uncertainty_method.endswith(suffix):
                predictor_kwargs["uncertainty_method"] = method
                break
        else:
            raise NotImplementedError

    detected_locations = []
    inference_times = {}

    tracker = EmissionsTracker(project_name=f"inference_{args.image_folders[0].stem}", measure_power_secs=10, log_level="error")
    tracker.start()

    img_files = get_image_paths(args.image_folders)

    for i in tqdm(range(0, len(img_files), args.batch_size), desc="Detecting"):
        batch = img_files[i : i + args.batch_size]

        # Set low conf threshold to save all detections
        predictions: list[CustomResults] = network.predict(
            batch, predictor=partial(predictor, **predictor_kwargs), imgsz=args.imgsz, conf=0.01
        )
        for prediction in predictions:
            with logging_redirect_tqdm():
                detections = prediction.to_detections()

                img_name = Path(prediction.path).stem
                camera = get_camera_by_name(chunk, img_name)

                border_coordinates = [
                    pixel_to_gps_location(chunk, camera, [0, 0]),
                    pixel_to_gps_location(chunk, camera, [0, camera.sensor.height]),
                    pixel_to_gps_location(chunk, camera, [camera.sensor.width, camera.sensor.height]),
                    pixel_to_gps_location(chunk, camera, [camera.sensor.width, 0]),
                ]
                fov_polygon = Polygon([l.to_point() for l in border_coordinates])

                for detection in detections:
                    location = pixel_to_gps_location(chunk, camera, detection.coordinate)
                    assert location is not None

                    location.properties["image_name"] = img_name
                    location.properties["class_name"] = detection.class_name
                    location.properties["confidence"] = detection.confidence
                    location.properties["image_location"] = detection.coordinate.tolist()
                    location.properties["image_size"] = [camera.sensor.width, camera.sensor.height]
                    location.properties["distance_to_border"] = fov_polygon.exterior.distance(location.to_point())

                    if detection.class_confidences is not None:
                        location.properties["class_confidences"] = detection.class_confidences

                    if detection.uncertainty_measure is not None:
                        location.properties["uncertainty_measure"] = detection.uncertainty_measure

                    detected_locations.append(location)

                inference_times[img_name] = prediction.speed

    tracker.stop()

    write_gps_locations_to_kml(output_file, detected_locations)

    inference_times_file = output_file.parent / (output_file.stem + "_times.json")
    with inference_times_file.open("w") as file_writer:
        dump(inference_times, file_writer, indent=2)


if __name__ == "__main__":
    main()
