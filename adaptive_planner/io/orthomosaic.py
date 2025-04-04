from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import TYPE_CHECKING

import numpy as np

from lxml import etree as ET
from rasterio import open as rasterio_open
from rasterio.io import MemoryFile
from rasterio.merge import merge
from shapely import Polygon

from adaptive_planner import setup_logging
from adaptive_planner.georeference import get_field_of_view_m, get_fov_polygon
from adaptive_planner.io.kml import get_element_value, get_namespace
from adaptive_planner.location import Location
from adaptive_planner.utils import CameraParameters, Waypoint

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from rasterio import DatasetReader

log = setup_logging(__name__)


def extract_roi_from_orthomosaic(
    orthomosaic_data: NDArray[np.uint8], roi_center: NDArray[np.uint16], size: NDArray[np.uint16], pad_value: int = 0
) -> NDArray[np.uint8]:
    roi = np.full((size[0], size[1], orthomosaic_data.shape[2]), pad_value, dtype=np.uint8)

    pad_size = np.maximum(size // 2 - roi_center, 0)
    roi_coordinates = np.concatenate(
        (np.maximum(roi_center - size // 2, 0), np.minimum(roi_center + size // 2, orthomosaic_data.shape[:2]))
    )

    roi[
        pad_size[0] : pad_size[0] + roi_coordinates[2] - roi_coordinates[0],
        pad_size[1] : pad_size[1] + roi_coordinates[3] - roi_coordinates[1],
        :,
    ] = orthomosaic_data[roi_coordinates[0] : roi_coordinates[2], roi_coordinates[1] : roi_coordinates[3], :]

    return roi


def get_tile_polygons(kml_file: Path) -> dict[Path, Polygon]:
    with kml_file.open("rb") as kml_file_buffer:
        kml_root = ET.fromstring(kml_file_buffer.read())

    namespace = {"kml": get_namespace(kml_root)}

    tile_boundaries: dict[Path, Polygon] = {}
    for placemark in kml_root.findall(".//kml:Placemark", namespaces=namespace):
        name = get_element_value(placemark, "kml:name", namespaces=namespace)
        coordinates = get_element_value(placemark, "kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", namespaces=namespace)

        gps_locations: list[Location] = []
        for coordinate in coordinates.split(" "):
            gps_locations.append(Location(np.fromstring(coordinate, dtype=np.float64, sep=",")))

        polygon = Polygon([p.utm_coordinate[:2] for p in gps_locations])

        tile_path = kml_file.parent / name
        if not tile_path.is_file():
            raise FileNotFoundError(f"Error: orthomosaic tile {tile_path.name} is missing!")

        tile_boundaries[tile_path] = polygon

    return tile_boundaries


def get_tiles_to_load(
    tiles: dict[Path, Polygon],
    waypoint: Waypoint,
    camera_parameters: CameraParameters,
) -> set[Path]:
    fov_m = get_field_of_view_m(waypoint.location.gps_coordinate_lon_lat[2], camera_parameters)
    fov_polygon = get_fov_polygon(waypoint, fov_m)

    tiles_to_load = []
    for tile_file, polygon in tiles.items():
        if polygon.intersects(fov_polygon):
            tiles_to_load.append(tile_file)

    return set(sorted(tiles_to_load))


def load_tiles(tiles_to_load: list[Path], parallel: bool = True) -> DatasetReader:
    log.info(f"Loading tiles {[t.name for t in tiles_to_load]}")
    _start_time = time()

    if len(tiles_to_load) == 1:
        return rasterio_open(tiles_to_load[0])

    if parallel:
        with ThreadPoolExecutor() as executor:
            loaded_datasets = list(executor.map(load_tile, tiles_to_load))
    else:
        loaded_datasets = [load_tile(tile) for tile in tiles_to_load]

    mosaic, output_trans = merge(loaded_datasets)
    output_meta = loaded_datasets[0].meta.copy()
    output_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output_trans,
        }
    )

    with MemoryFile() as memfile:
        with memfile.open(**output_meta) as writer:
            writer.write(mosaic)

        log.info(f"Loaded tiles in {time() - _start_time:.2f} seconds")

        return rasterio_open(memfile)


def load_tile(tile_path: Path) -> DatasetReader:
    with rasterio_open(tile_path) as src:
        with MemoryFile() as memfile:
            with memfile.open(
                driver=src.driver,
                width=src.width,
                height=src.height,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=src.transform,
            ) as dataset:
                dataset.write(src.read())

            return rasterio_open(memfile)
