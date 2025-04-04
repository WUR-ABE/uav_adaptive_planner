from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copy
from tap import Tap
from typing import TYPE_CHECKING, Literal

from adaptive_planner import setup_logging
from adaptive_planner.executor.orthomosaic import OrthomosaicSim
from adaptive_planner.executor.simple import SimpleSim
from adaptive_planner.io.kml import read_kml_file
from adaptive_planner.planner import AdaptivePlanner
from adaptive_planner.tracking import GPSTrack, TrackerStatus

if TYPE_CHECKING:
    from adaptive_planner.executor import Executor

EXECUTORS: dict[str, type[Executor]] = {
    "orthomosaic_sim": OrthomosaicSim,
    "simple_sim": SimpleSim,
}

log = setup_logging(__name__)


class PlannerException(Exception):
    pass


def load_base_flight(base_flight_folder: Path, adaptive_planner: AdaptivePlanner) -> None:
    detections_file = base_flight_folder / "detections.kml"
    flight_path_file = base_flight_folder / "flight_path.kml"

    if not detections_file.is_file() or not flight_path_file.is_file():
        raise FileNotFoundError("Detections file or flight path file does not exist!")

    detections = read_kml_file(detections_file)
    flight_path = read_kml_file(flight_path_file)

    for detection in detections:
        tracker = GPSTrack.from_location(
            detection,
            rejection_confidence=adaptive_planner.config.rejection_confidence,
            inspection_confidence=adaptive_planner.config.inspection_confidence,
        )

        if tracker is not None:
            adaptive_planner.tracker.add_tracker(tracker)

    adaptive_planner._executed_flight_path = flight_path

    if adaptive_planner.output_folder is not None:
        adaptive_planner.save_config()
        base_flight_path_image = base_flight_folder / "base_flight_path.png"
        if base_flight_path_image.is_file():
            copy(base_flight_path_image, adaptive_planner.output_folder)


class ArgumentParser(Tap):
    executor: Literal["orthomosaic_sim", "simple_sim"]  # Executor to use
    field_file: Path  # Path to the field config file
    config_file: Path  # Path to the planner config file
    executor_config_file: Path | None = None  # Path to the executor config file
    base_flight_folder: Path | None = None  # Path to the base fligt path to reuse
    output_folder: Path = Path(f"evaluation_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")  # Output folder to save results
    gui: bool = False  # Show user interface

    def configure(self) -> None:
        self.add_argument("executor", choices=EXECUTORS.keys())
        self.add_argument("field_file")
        self.add_argument("config_file")

    def process_args(self) -> None:
        if not self.field_file.is_file():
            raise FileNotFoundError(f"Field file {self.field_file.name} does not exist!")

        if not self.config_file.is_file():
            raise FileNotFoundError(f"Config file {self.config_file.name} does not exist!")

        if self.executor_config_file is not None and not self.executor_config_file.is_file():
            raise FileNotFoundError(f"Executor config file {self.executor_config_file.name} does not exist!")

        if self.base_flight_folder is not None and not self.base_flight_folder.is_dir():
            raise FileNotFoundError(f"Base flight folder {self.base_flight_folder.name} does not exist!")


def main() -> None:
    from yaml import safe_load

    from adaptive_planner.field import Field
    from adaptive_planner.gui import AdaptivePlannerGui
    from adaptive_planner.io.kml import write_gps_flightpath_to_kml, write_gps_locations_to_kml
    from adaptive_planner.planner import AdaptivePlannerConfig, InvestigationPlanningStrategy
    from adaptive_planner.planners.coverage import CoveragePlanner, CoveragePlannerConfig
    from adaptive_planner.planners.tsp import TSPPlanner, TSPPlannerConfig

    args = ArgumentParser().parse_args()

    field = Field.from_file(args.field_file)

    with args.config_file.open("r") as config_file_handler:
        config = safe_load(config_file_handler)

    coverage_planner_config = CoveragePlannerConfig(**config["base_planner"])
    tsp_planner_config = TSPPlannerConfig(**config["inspection_planner"])
    adaptive_planner_config = AdaptivePlannerConfig(**config["adaptive_planner"])

    executor = (
        EXECUTORS[args.executor](field, args.output_folder)  # type: ignore [call-arg]
        if args.executor_config_file is None
        else EXECUTORS[args.executor].from_file(field, args.output_folder, args.executor_config_file)
    )
    coverage_planner = CoveragePlanner(field, executor.camera_parameters, coverage_planner_config)
    tsp_planner = TSPPlanner(field, executor.camera_parameters, tsp_planner_config)
    adaptive_planner = AdaptivePlanner(
        field,
        executor.camera_parameters,
        coverage_planner,
        tsp_planner,
        executor.get_image_at_coordinate,
        config=adaptive_planner_config,
        output_folder=args.output_folder,
    )

    if args.base_flight_folder is not None:
        if adaptive_planner_config.planning_strategy != InvestigationPlanningStrategy.AFTERWARDS:
            raise PlannerException("Can only load base flight folder when using planning strategy afterwards!")

        load_base_flight(args.base_flight_folder, adaptive_planner)

    if args.gui:
        # GUI also takes care of saving results
        app = AdaptivePlannerGui(executor, field, adaptive_planner)
        adaptive_planner.set_callbacks(
            drone_waypoint_callback=app.update_drone_waypoint,
            base_flight_path_callback=app.update_base_flight_path,
            detection_image_callback=app.update_detection_image,
            detection_callback=app.add_detection,
        )

        executor.enable()

        app.start_task(only_inspection=args.base_flight_folder is not None)
        app.mainloop()
    else:
        executor.enable()

        if args.base_flight_folder is None:
            adaptive_planner.execute()
        else:
            adaptive_planner.execute_inspection(
                adaptive_planner.tracker.get_locations(status_mask=TrackerStatus.TO_BE_INVESTIGATED),
                adaptive_planner.executed_flight_path[-1],
                None,
            )

        if adaptive_planner.output_folder is not None:
            write_gps_flightpath_to_kml(
                adaptive_planner.output_folder / "flight_path.kml",
                adaptive_planner.executed_flight_path,
                f"{adaptive_planner.output_folder} flight path",
            )
            log.info(f"Saved flight path to {adaptive_planner.output_folder / 'flight_path.kml'}")

            write_gps_locations_to_kml(adaptive_planner.output_folder / "detections.kml", adaptive_planner.detected_objects)
            log.info(f"Saved detected objects to {adaptive_planner.output_folder / 'detections.kml'}")

    executor.finish()


if __name__ == "__main__":
    main()
