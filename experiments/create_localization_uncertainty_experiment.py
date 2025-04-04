from __future__ import annotations

from pathlib import Path
from tap import Tap
from typing import Literal

CONFIG_CONTENT = """base_planner:
  sideways_overlap: 0.1
  forward_overlap: 0.1

inspection_planner:
  max_tsp_calculation_time: 2.0
  
adaptive_planner:
  weights_file: adaptive_planner/best_n.pt

  base_altitude: {alt}.0
  min_inspection_altitude: 12.0
  max_inspection_altitude: 12.0

  inspection_confidence: {accept_threshold}
  rejection_confidence: {reject_threshold}

  # Detection network parameters
  imgsz: 2048
  iou: 0.2
  agnostic_nms: true  # E.g. different classes won't overlap

  distance_threshold: {distance_threshold}

  planning_strategy: AFTERWARDS
  use_adaptive: {adaptive}
  use_tqdm: true
"""

EXECUTOR_CONTENT = """position_uncertainty_m: {position_uncertainty:.2f}
altitude_uncertainty_m: {altitude_uncertainty:.2f}
heading_uncertainty_deg: {heading_uncertainty}
roll_uncertainty_deg: {roll_uncertainty}
pitch_uncertainty_deg: {pitch_uncertainty}

seed: -1
"""


class ArgumentError(Exception):
    pass


class ArgumentParser(Tap):
    experiment_folder: Path  # Path to the experiment folder
    level_name: str  # Name of the uncertainty level
    plant_distribution: Literal["uniform", "clustered"]  # Distribution to use
    accept_threshold: float  # Best acceptance threshold
    reject_threshold: float  # Best rejection threshold
    position_uncertainty: float  # Uncertainty in position
    altitude_uncertainty: float  # Uncertainty in altitude
    heading_uncertainty: float  # Uncertainty in heading
    roll_uncertainty: float  # Uncertainty in roll
    pitch_uncertainty: float  # Uncertainty in pitch
    distance_thresholds: float  # Maximum distance between detections
    altitudes: list[int]  # Range of altitudes

    def configure(self) -> None:
        self.add_argument("experiment_folder")
        self.add_argument("level_name")
        self.add_argument("--altitudes", nargs="+")


def main() -> None:
    args = ArgumentParser().parse_args()

    if not args.experiment_folder.is_dir():
        args.experiment_folder.mkdir(parents=True)

    for alt in args.altitudes:
        filename = args.experiment_folder / f"{alt}m_planner_{args.plant_distribution}_{args.level_name}_config.yaml"
        filename.write_text(
            CONFIG_CONTENT.format(
                alt=alt,
                accept_threshold=args.accept_threshold,
                reject_threshold=args.reject_threshold,
                distance_threshold=args.distance_thresholds,
                adaptive="true",
            )
        )

        filename = args.experiment_folder / f"baseline_{alt}m_planner_{args.plant_distribution}_{args.level_name}_config.yaml"
        filename.write_text(
            CONFIG_CONTENT.format(
                alt=alt,
                accept_threshold=args.accept_threshold,
                reject_threshold=args.reject_threshold,
                distance_threshold=args.distance_thresholds,
                adaptive="false",
            )
        )

    filename = args.experiment_folder / f"{args.level_name}.yaml"
    filename.write_text(
        EXECUTOR_CONTENT.format(
            position_uncertainty=args.position_uncertainty,
            altitude_uncertainty=args.altitude_uncertainty,
            heading_uncertainty=args.heading_uncertainty,
            roll_uncertainty=args.roll_uncertainty,
            pitch_uncertainty=args.pitch_uncertainty,
        )
    )


if __name__ == "__main__":
    main()

    # create_localization_uncertainty_experiment experiments/localization_uncertainty perfect --plant_distribution clustered --accept_threshold 0.6 --reject_threshold 0.05 --position_uncertainty 0.0 --altitude_uncertainty 0.0 --heading_uncertainty 0.0 --roll_uncertainty 0.0 --pitch_uncertainty 0.0 --distance_thresholds 0.35 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty good --plant_distribution clustered --accept_threshold 0.6 --reject_threshold 0.05 --position_uncertainty 0.015 --altitude_uncertainty 0.015 --heading_uncertainty 0.25 --roll_uncertainty 0.25 --pitch_uncertainty 0.25 --distance_thresholds 0.6 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty decent --plant_distribution clustered --accept_threshold 0.6 --reject_threshold 0.05 --position_uncertainty 0.03 --altitude_uncertainty 0.03 --heading_uncertainty 0.5 --roll_uncertainty 0.5 --pitch_uncertainty 0.5 --distance_thresholds 0.8 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty poor --plant_distribution clustered --accept_threshold 0.6 --reject_threshold 0.05 --position_uncertainty 0.10 --altitude_uncertainty 0.10 --heading_uncertainty 1.0 --roll_uncertainty 1.0 --pitch_uncertainty 1.0 --distance_thresholds 1.3 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty very_poor --plant_distribution clustered --accept_threshold 0.6 --reject_threshold 0.05 --position_uncertainty 0.20 --altitude_uncertainty 0.20 --heading_uncertainty 2.0 --roll_uncertainty 2.0 --pitch_uncertainty 2.0 --distance_thresholds 2.3 --altitudes 12 24 36 48

    # create_localization_uncertainty_experiment experiments/localization_uncertainty perfect --plant_distribution uniform --accept_threshold 0.8 --reject_threshold 0.05 --position_uncertainty 0.0 --altitude_uncertainty 0.0 --heading_uncertainty 0.0 --roll_uncertainty 0.0 --pitch_uncertainty 0.0 --distance_thresholds 0.35 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty good --plant_distribution uniform --accept_threshold 0.8 --reject_threshold 0.05 --position_uncertainty 0.015 --altitude_uncertainty 0.015 --heading_uncertainty 0.25 --roll_uncertainty 0.25 --pitch_uncertainty 0.25 --distance_thresholds 0.6 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty decent --plant_distribution uniform --accept_threshold 0.8 --reject_threshold 0.05 --position_uncertainty 0.03 --altitude_uncertainty 0.03 --heading_uncertainty 0.5 --roll_uncertainty 0.5 --pitch_uncertainty 0.5 --distance_thresholds 0.8 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty poor --plant_distribution uniform --accept_threshold 0.8 --reject_threshold 0.05 --position_uncertainty 0.10 --altitude_uncertainty 0.10 --heading_uncertainty 1.0 --roll_uncertainty 1.0 --pitch_uncertainty 1.0 --distance_thresholds 1.3 --altitudes 12 24 36 48
    # create_localization_uncertainty_experiment experiments/localization_uncertainty very_poor --plant_distribution uniform --accept_threshold 0.8 --reject_threshold 0.05 --position_uncertainty 0.20 --altitude_uncertainty 0.20 --heading_uncertainty 2.0 --roll_uncertainty 2.0 --pitch_uncertainty 2.0 --distance_thresholds 2.3 --altitudes 12 24 36 48
