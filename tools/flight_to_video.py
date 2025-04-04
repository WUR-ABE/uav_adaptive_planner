from __future__ import annotations

from pathlib import Path
from tap import Tap

from adaptive_planner import setup_logging
from adaptive_planner.io.kml import read_kml_file
from adaptive_planner.visualisation import plot_animated_flight_path, save_animation_with_progress_bar

log = setup_logging(__name__)


class ArgumentParser(Tap):
    input_folder: Path  # Path to the input folder
    output_file: Path | None = None  # Output .mp4 or .gif file
    interpolation_steps: float = 1.0  # Render distance of flight path in meters
    flight_steps_per_second: int = 40  # Speed of the video in m/s flight path
    overwrite: bool = False  # Overwrite output file if exists
    no_colorbar: bool = False  # Disable the color bar

    def configure(self) -> None:
        self.add_argument("input_folder")


def main() -> None:
    args = ArgumentParser().parse_args()

    flight_path_file = args.input_folder
    if flight_path_file.is_dir():
        flight_path_file = args.input_folder / "flight_path.kml"

    if not flight_path_file.is_file():
        raise FileNotFoundError(f"Flight path file '{flight_path_file}' does not exist!")

    output_file: Path | None = args.output_file
    if output_file is None:
        output_file = flight_path_file.parent / (flight_path_file.parent.name + ".mp4")

    if output_file.is_file() and args.overwrite:
        output_file.unlink()
    elif output_file.is_file():
        raise FileExistsError(f"Output file '{output_file}' already exists!")

    flight_path = read_kml_file(flight_path_file)

    animation = plot_animated_flight_path(flight_path, interpolation_steps_m=args.interpolation_steps, colorbar=not args.no_colorbar)
    save_animation_with_progress_bar(animation, output_file, fps=round(args.flight_steps_per_second / args.interpolation_steps))

    log.info(f"Saved video as '{output_file}'")


if __name__ == "__main__":
    main()
