from __future__ import annotations

from pathlib import Path
from tap import Tap
from typing import TYPE_CHECKING

from shapely import Polygon
from tqdm.auto import tqdm

from adaptive_planner import logging_redirect_tqdm, setup_logging
from adaptive_planner.io.kml import write_gps_locations_to_kml
from adaptive_planner.io.topcon import read_topcon_data

from .metashape import get_visible_locations, gps_to_pixel_coordinate, load_project, pixel_to_gps_location

if TYPE_CHECKING:
    from adaptive_planner.location import Location

log = setup_logging(__name__)


class ArgumentParser(Tap):
    agisoft_project_file: Path  # Path to the Agisoft Metashape project (*.psx) file
    gt_file: Path  # Path to the GT file
    output_file: Path  # Output kml file
    chunk_name: str | None = None  # Name of the chunk in Agisoft
    gt_file_crs: str = "epsg:28992"  # Input crs of the gt file

    def configure(self) -> None:
        self.add_argument("agisoft_project_file")
        self.add_argument("gt_file")
        self.add_argument("output_file")

    def process_args(self) -> None:
        if not self.agisoft_project_file.is_file():
            raise FileNotFoundError(f"Agisoft project file {self.agisoft_project_file.name} does not exist!")

        if not self.gt_file.is_file():
            raise FileNotFoundError(f"GT file {self.gt_file.name} does not exist!")


def main() -> None:
    args = ArgumentParser().parse_args()

    chunk = load_project(args.agisoft_project_file, chunk_name=args.chunk_name)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    if chunk is None:
        raise RuntimeError("Could not load Agisoft project")

    gt_locations = read_topcon_data(args.gt_file, crs=args.gt_file_crs, use_height=True)

    gt_locations_per_image: list[Location] = []
    for camera in tqdm(chunk.cameras):
        border_coordinates = [
            pixel_to_gps_location(chunk, camera, [0, 0]),
            pixel_to_gps_location(chunk, camera, [0, camera.sensor.height]),
            pixel_to_gps_location(chunk, camera, [camera.sensor.width, camera.sensor.height]),
            pixel_to_gps_location(chunk, camera, [camera.sensor.width, 0]),
        ]
        fov_polygon = Polygon([l.to_point() for l in border_coordinates])

        with logging_redirect_tqdm():
            for l in get_visible_locations(chunk, camera, gt_locations):
                location = l.copy()
                location.properties["name"] = f"object_{len(gt_locations_per_image)}"
                location.properties["gt_name"] = l.properties["name"]
                location.properties["image_name"] = camera.label
                location.properties["image_location"] = gps_to_pixel_coordinate(chunk, camera, location).tolist()
                location.properties["image_size"] = [camera.sensor.width, camera.sensor.height]
                location.properties["distance_to_border"] = fov_polygon.exterior.distance(location.to_point())
                gt_locations_per_image.append(location)

    write_gps_locations_to_kml(args.output_file, gt_locations_per_image)


if __name__ == "__main__":
    main()
