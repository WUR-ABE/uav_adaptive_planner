from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from json import dump
from logging import FileHandler, Formatter, getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tqdm import tqdm
from ultralytics import YOLO

from adaptive_planner import logging_redirect_tqdm, setup_logging
from adaptive_planner.field import Field
from adaptive_planner.location import Location
from adaptive_planner.predict import CustomPredictor, Detection
from adaptive_planner.tracking import GPSTracker, TrackerStatus
from adaptive_planner.utils import Waypoint

if TYPE_CHECKING:
    from typing import Any, Callable

    from numpy.typing import NDArray

    from adaptive_planner.planners import Planner
    from adaptive_planner.predict import CustomResults
    from adaptive_planner.utils import CameraParameters


class InvestigationPlanningStrategy(Enum):
    BETWEEN_WAYPOINTS = 0
    AFTERWARDS = 1


@dataclass
class AdaptivePlannerConfig:
    weights_file: Path = Path("adaptive_planner/best_n.pt")

    base_altitude: float = 20.0
    min_inspection_altitude: float = 12.0
    max_inspection_altitude: float = 20.0

    inspection_confidence: float = 0.8
    rejection_confidence: float = 0.05

    # Detection network parameters
    imgsz: int = 2048
    iou: float = 0.2
    agnostic_nms: bool = True  # E.g. different classes won't overlap

    distance_threshold: dict[float, float] | float = 0.35

    use_adaptive: bool = True
    planning_strategy: InvestigationPlanningStrategy = InvestigationPlanningStrategy.BETWEEN_WAYPOINTS
    use_tqdm: bool = False
    save_detection_img: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.weights_file, str):
            self.weights_file = Path(self.weights_file)

        if isinstance(self.planning_strategy, str):
            self.planning_strategy = InvestigationPlanningStrategy[self.planning_strategy.upper()]

        assert self.weights_file.is_file()

    def as_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["weights_file"] = str(self.weights_file)
        d["planning_strategy"] = self.planning_strategy.name
        return d


log = setup_logging(__name__)


class AdaptivePlanner:
    def __init__(
        self,
        field: Field,
        camera_parameters: CameraParameters,
        base_flight_path_planner: Planner,
        inspection_fligth_path_planner: Planner,
        executor_callback: Callable[[Waypoint], NDArray[np.uint8]],
        output_folder: Path | None = None,
        config: AdaptivePlannerConfig = AdaptivePlannerConfig(),
    ) -> None:
        self.field = field
        self.executor_callback = executor_callback

        self.base_flight_path_planner = base_flight_path_planner
        self.inspection_flight_path_planner = inspection_fligth_path_planner
        self.config = config

        self.detection_network = YOLO(self.config.weights_file)
        self._base_fligth_path: list[Waypoint] | None = None

        self.output_folder = output_folder
        if self.output_folder is not None and not self.output_folder.is_dir():
            log.info(f"Created output folder '{self.output_folder}'")
            self.output_folder.mkdir(parents=True)

        self._camera_parameters = camera_parameters

        self._executed_flight_path: list[Location] = []

        self.setup_file_logging()

        self.tracker = GPSTracker(
            self.field,
            self._camera_parameters,
            distance_threshold=self.config.distance_threshold,
            max_inspection_altitude=self.config.max_inspection_altitude,
            use_adaptive=self.config.use_adaptive,
        )

        self.drone_waypoint_callback: Callable[[Waypoint, list[Location]], None] | None = None
        self.base_flight_path_callback: Callable[[list[Waypoint]], None] | None = None
        self.detection_image_callback: Callable[[NDArray[np.uint8]], None] | None = None
        self.detection_callback: Callable[[Location], None] | None = None

    @property
    def executed_flight_path(self) -> list[Location]:
        return self._executed_flight_path

    @property
    def detected_objects(self) -> list[Location]:
        return self.tracker.get_locations()

    def set_callbacks(
        self,
        drone_waypoint_callback: Callable[[Waypoint, list[Location]], None] | None = None,
        base_flight_path_callback: Callable[[list[Waypoint]], None] | None = None,
        detection_image_callback: Callable[[NDArray[np.uint8]], None] | None = None,
        detection_callback: Callable[[Location], None] | None = None,
    ) -> None:
        self.drone_waypoint_callback = drone_waypoint_callback
        self.base_flight_path_callback = base_flight_path_callback
        self.detection_image_callback = detection_image_callback
        self.detection_callback = detection_callback

    def reset(self) -> None:
        self._executed_flight_path = []

    def execute(self) -> None:
        self.save_config()

        start_location = self.field.start_location
        if start_location is None:
            log.warning("Start location is not defined, calculating location from field boundary!")
            coordinates = np.array(self.field.boundary.exterior.coords[0], dtype=np.float64)
            start_location = Location(coordinates)

        if self._base_fligth_path is None:
            output_file = self.output_folder / "base_flight_path.png" if self.output_folder else None
            self._base_fligth_path = self.base_flight_path_planner.plan(start_location, self.config.base_altitude, output_file=output_file)

            if self.base_flight_path_callback is not None:
                self.base_flight_path_callback(self._base_fligth_path)

        # Send start position
        if self.drone_waypoint_callback is not None:
            self.drone_waypoint_callback(Waypoint(start_location, 0.0), self._executed_flight_path)

        log.info(f"Loading YOLO weights from {self.config.weights_file.name}...")

        for i in tqdm(range(len(self._base_fligth_path)), desc="Base flight path", disable=not self.config.use_tqdm):
            with logging_redirect_tqdm():
                waypoint = self._base_fligth_path[i]

                log.info(f"Base flight path waypoint {i}")

                # waypoint.location.gps_coordinate_lon_lat[0] = 5.668272950299547
                # waypoint.location.gps_coordinate_lon_lat[1] = 51.99146048181497
                # waypoint.location.gps_coordinate_lon_lat[2] = 16.0

                self.investigate_waypoint(waypoint)

                # Send callback when needed
                if self.detection_callback:
                    for loc in self.tracker.get_locations(status_mask=TrackerStatus.ACCEPTED | TrackerStatus.INVESTIGATED):
                        self.detection_callback(loc)

                if self.config.planning_strategy == InvestigationPlanningStrategy.BETWEEN_WAYPOINTS:
                    self.execute_inspection(
                        self.tracker.get_locations(status_mask=TrackerStatus.TO_BE_INVESTIGATED),
                        waypoint.location,
                        (None if i >= len(self._base_fligth_path) - 1 else self._base_fligth_path[i + 1].location),
                        inspection_name=f"base_waypoint_{i}",
                    )

        if self.config.planning_strategy == InvestigationPlanningStrategy.AFTERWARDS:
            self.execute_inspection(
                self.tracker.get_locations(status_mask=TrackerStatus.TO_BE_INVESTIGATED),
                waypoint.location,
                None,
            )

    def execute_inspection(
        self,
        locations: list[Location],
        start_location: Location,
        end_location: Location | None,
        inspection_name: str = "",
    ) -> None:
        inspection_waypoints = self.create_inspection_waypoints(locations)

        if len(inspection_waypoints) > 0:
            log.info(f"Got {len(inspection_waypoints)} waypoints for furthur investigation")

            self.inspection_flight_path_planner.set_objects_and_end_location(inspection_waypoints, end_location)

            name = "inspection_flight_path.png" if inspection_name == "" else f"inspection_flight_path_{inspection_name}.png"
            output_file = self.output_folder / name if self.output_folder else None
            inspection_flight_path = self.inspection_flight_path_planner.plan(
                start_location, self.config.min_inspection_altitude, output_file=output_file
            )

            for inspection_waypoint in tqdm(
                inspection_flight_path, desc="Inspection flight path", disable=not self.config.use_tqdm, leave=False
            ):
                self.investigate_waypoint(inspection_waypoint)

                # Send callback when needed
                if self.detection_callback:
                    for loc in self.tracker.get_locations(status_mask=TrackerStatus.ACCEPTED | TrackerStatus.INVESTIGATED):
                        self.detection_callback(loc)
        else:
            log.info("Got no objects for furthur investigation")

    def investigate_waypoint(self, waypoint: Waypoint) -> None:
        """
        This function let the drone fly to a waypoint, classifies the image and gives a list of
        waypoints for further investigation.
        """
        with np.printoptions(precision=15):
            log.info(f"Flying to location={waypoint.location} heading={waypoint.heading}")

        image = self.executor_callback(waypoint)
        self._executed_flight_path.append(waypoint.location)

        if self.drone_waypoint_callback is not None:
            self.drone_waypoint_callback(waypoint, self._executed_flight_path)

        detections = self.detect(image)
        self.tracker.track(waypoint, detections)

    def create_inspection_waypoints(self, locations: list[Location]) -> list[Location]:
        filtered_locations = []

        for location in locations:
            location = location.copy()
            best_confidence = max([conf for conf in location.properties["confidences"] if conf is not None])

            # Calculate the optimal height proportional to the confidence
            conf_norm = (best_confidence - self.config.rejection_confidence) / (
                self.config.inspection_confidence - self.config.rejection_confidence
            )
            altitude = (
                self.config.max_inspection_altitude - self.config.min_inspection_altitude
            ) * conf_norm + self.config.min_inspection_altitude

            # Skip when we already have seen the tracker lower than this altitude, inspection then makes no sense
            if altitude >= min(location.properties["drone_altitudes"]):
                continue

            # Update altitude
            if location.has_altitude:
                location.gps_coordinate_lon_lat[2] = altitude
            else:
                location.gps_coordinate_lon_lat = np.array([*location.gps_coordinate_lon_lat[:2], altitude], dtype=np.float64)

            log.info(f"Adding object {location.properties['name']} to list for furthur inspection...")
            filtered_locations.append(location)

        return filtered_locations

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        image = image[:, :, ::-1]  # RGB -> BGR

        # TODO: add parameters
        yolo_prediction: CustomResults = self.detection_network.predict(
            image,
            predictor=CustomPredictor,
            conf=self.config.rejection_confidence,
            imgsz=self.config.imgsz,
            iou=self.config.iou,
            agnostic_nms=self.config.agnostic_nms,
            verbose=False,
        )[0]

        yolo_prediction = yolo_prediction.cpu()

        if self.config.save_detection_img and self.output_folder is not None:
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            yolo_prediction.plot(save=True, filename=self.output_folder / f"detection_img_{current_time_str}.jpg")

        if self.detection_image_callback is not None:
            self.detection_image_callback(yolo_prediction.plot())

        log.info(f"Found {len(yolo_prediction.boxes)} detection(s)")

        return yolo_prediction.to_detections()

    def setup_file_logging(self) -> None:
        # Add filehandler to logger, to be sure that we log stuff if something fails
        if self.output_folder:
            log_file_handler = FileHandler(self.output_folder / "log.txt")
            formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
            log_file_handler.setFormatter(formatter)
            getLogger().addHandler(log_file_handler)

    def save_config(self) -> None:
        if self.output_folder:
            with (self.output_folder / "config.json").open("w") as config_file_handler:
                config = self.config.as_dict()
                config["base_planner"] = self.base_flight_path_planner.config_dict
                config["inspection_planner"] = self.inspection_flight_path_planner.config_dict
                dump(config, config_file_handler, indent=2)
