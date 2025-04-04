from __future__ import annotations

from json import dump, load
from typing import TYPE_CHECKING

import numpy as np

from shapely import Polygon

from adaptive_planner.location import WGS_84, Location

if TYPE_CHECKING:
    from pathlib import Path

    from pyproj import CRS


def write_polygon_file(output_file: Path, boundary: Polygon, name: str, polygon_crs: CRS = WGS_84) -> None:
    points_gps = [Location.from_crs(np.array(p, dtype=np.float64), polygon_crs) for p in boundary.exterior.coords]

    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"Name": name, "Description": ""},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[list(p.gps_coordinate_lon_lat[:2]) for p in points_gps]],
                },
            }
        ],
    }

    with output_file.open("w") as file_stream:
        dump(data, file_stream, indent=2)


def read_polygon_file(file: Path, polygon_crs: CRS = WGS_84) -> Polygon:
    with file.open("r") as file_stream:
        data = load(file_stream)

        for feature in data["features"]:
            if feature["geometry"]["type"] != "Polygon":
                continue

            coordinates = np.stack([Location(c).to_crs(polygon_crs) for c in feature["geometry"]["coordinates"][0]], axis=0)
            return Polygon(coordinates.squeeze())

    raise RuntimeError("Polygon geojson has not expected format!")
