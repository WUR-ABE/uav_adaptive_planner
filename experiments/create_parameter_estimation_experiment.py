from __future__ import annotations

from pathlib import Path
from tap import Tap

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
  rejection_confidence: {reject_treshold}

  # Detection network parameters
  imgsz: 2048
  iou: 0.2
  agnostic_nms: true  # E.g. different classes won't overlap

  distance_threshold: 0.35

  planning_strategy: AFTERWARDS
  use_adaptive: {adaptive}
  use_tqdm: true
"""


class ArgumentParser(Tap):
    experiment_folder: Path  # Path to the experiment folder
    altitudes: list[int]  # Range of altitudes
    accept_thresholds: list[float]  # Range of accept thresholds
    reject_thresholds: list[float]  # Range of reject thresholds

    def configure(self) -> None:
        self.add_argument("experiment_folder")
        self.add_argument("--altitudes", nargs="+")
        self.add_argument("--accept_thresholds", nargs="+")
        self.add_argument("--reject_thresholds", nargs="+")


def main() -> None:
    args = ArgumentParser().parse_args()

    if not args.experiment_folder.is_dir():
        args.experiment_folder.mkdir(parents=True)

    for alt in args.altitudes:
        for accept_threshold in args.accept_thresholds:
            for reject_treshold in args.reject_thresholds:
                if accept_threshold == reject_treshold:
                    continue

                filename = args.experiment_folder / f"alt_{alt}m_accept_{accept_threshold}_reject_{reject_treshold}.yaml"
                filename.write_text(
                    CONFIG_CONTENT.format(alt=alt, accept_threshold=accept_threshold, reject_treshold=reject_treshold, adaptive="true")
                )

        filename = args.experiment_folder / f"baseline_{alt}m.yaml"
        filename.write_text(CONFIG_CONTENT.format(alt=alt, accept_threshold=0.8, reject_treshold=0.05, adaptive="false"))


if __name__ == "__main__":
    main()

    # create_parameter_estimation_experiment experiments/parameter_estimation --altitudes 12 24 36 48 --accept_thresholds 1.0 0.8 0.6 0.4 --reject_thresholds 0.05 0.2 0.4
