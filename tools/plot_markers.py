from __future__ import annotations

from pathlib import Path
from tap import Tap

import matplotlib.pyplot as plt

from adaptive_planner.io.topcon import read_topcon_data
from adaptive_planner.visualisation import plot_markers


class ArgumentParser(Tap):
    markers_file: Path  # GPS markers file
    output_file: Path | None = None  # Path to the output file
    markers_file_crs: str = "epsg:28992"  # Input crs of the markers file

    def configure(self) -> None:
        self.add_argument("markers_file")

    def process_args(self) -> None:
        if not self.markers_file.is_file():
            raise FileNotFoundError(f"Input file {self.markers_file.name} does not exist!")


def main() -> None:
    args = ArgumentParser().parse_args()

    markers = read_topcon_data(args.markers_file, crs=args.markers_file_crs)

    _, ax = plt.subplots(figsize=(10, 6), dpi=300)

    class_color_map = {
        "W": (97, 150, 202),
        "F": (199, 86, 106),
    }
    plot_markers(markers, class_color_map, ax=ax)

    plt.tight_layout()
    plt.show() if args.output_file is None else plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
