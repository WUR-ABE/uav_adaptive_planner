from __future__ import annotations

from pathlib import Path
from tap import Tap

from adaptive_planner import setup_logging
from adaptive_planner.io.topcon import read_topcon_data, write_topcon_data

log = setup_logging("transform_markers")


class ArgumentParser(Tap):
    markers_file: Path  # Input file
    output_file: Path  # Output file file
    output_file_crs: str  # EPSG code of output file
    markers_file_crs: str = "epsg:28992"  # Input crs of the markers file

    def configure(self) -> None:
        self.add_argument("markers_file")
        self.add_argument("output_file")
        self.add_argument("output_file_crs")

    def process_args(self) -> None:
        if not self.markers_file.is_file():
            raise FileNotFoundError(f"Input file {self.markers_file.name} does not exist!")


def main() -> None:
    args = ArgumentParser().parse_args()

    marker_locations = read_topcon_data(args.markers_file, crs=args.markers_file_crs, use_height=True)
    write_topcon_data(marker_locations, args.output_file, crs=args.output_file_crs)


if __name__ == "__main__":
    main()
