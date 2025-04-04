from __future__ import annotations

from pathlib import Path
from tap import Tap

from adaptive_planner.draw_field import DrawFieldGUI


class ArgumentParser(Tap):
    output_folder: Path | None = None  # Output folder for the field files
    name: str = "field"  # Name for the output field
    scheme_file: Path | None = None  # Optional scheme file to show borders of orthomosiac
    map_position: tuple[float, float] = (5.66819, 51.99140)  # Start position of map (longitude, latitude)

    def configure(self) -> None:
        self.add_argument("--map_position", nargs=2)


def main() -> None:
    args = ArgumentParser().parse_args()

    gui = DrawFieldGUI(args.map_position, scheme_file=args.scheme_file)
    gui.mainloop()

    field = gui.to_field(folder=args.output_folder, name=args.name, scheme_file=args.scheme_file)
    print(field)


if __name__ == "__main__":
    main()
