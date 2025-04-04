from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import rasterio
from rasterio.crs import CRS
from rasterio.transform import rowcol, xy
from shapely import Polygon

from adaptive_planner.location import Location
from adaptive_planner.utils import Annotation, CameraParameters, Waypoint

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from affine import Affine

    from adaptive_planner.predict import Detection


def get_orthomosaic_resolution(transform: Affine) -> NDArray[np.float64]:
    resolution_x = (transform[0] ** 2 + transform[1] ** 2) ** 0.5
    resolution_y = (transform[3] ** 2 + transform[4] ** 2) ** 0.5
    return np.array([resolution_x, resolution_y], dtype=np.float64)


def orthomosaic_gps_to_annotation(
    orthomosaic_tile: Path, gps_lon_lat: NDArray[np.float64], class_name: str, annotation_size: NDArray[np.float64]
) -> Annotation | None:
    with rasterio.open(orthomosaic_tile) as raster:
        if raster.crs != CRS.from_epsg(4326):
            print("Error: Orthomosaic should be in EPSG:4326 (WGS84).")
            return None

        y, x = rowcol(raster.transform, *gps_lon_lat)

        annotation_size_px = annotation_size / get_orthomosaic_resolution(raster.transform)

        return Annotation(
            "",
            max(0, x - annotation_size_px[0] // 2),
            max(0, y - annotation_size_px[1] // 2),
            annotation_size_px[0],
            annotation_size_px[1],
            class_name,
            orthomosaic_tile.stem,
            np.array([raster.width, raster.height], dtype=np.uint16),
        )


def orthomosaic_annotation_to_gps(orthomosaic_tile: Path, annotation: Annotation) -> NDArray[np.float64]:
    with rasterio.open(orthomosaic_tile) as raster:
        if raster.crs != CRS.from_epsg(4326):
            raise RuntimeError("Error: Orthomosaic should be in EPSG:4326 (WGS84).")

        tl_xs, tl_ys = xy(raster.transform, annotation.coordinate[1], annotation.coordinate[0])
        # br_xs, br_ys = xy(raster.transform, annotation.x + annotation.w, annotation.y + annotation.h)

    return np.array([tl_ys, tl_xs, np.nan], dtype=np.float64)


def get_field_of_view_m(altitude: float, camera_parameters: CameraParameters) -> NDArray[np.float64]:
    return (np.array(camera_parameters.sensor_size_mm, dtype=np.float64) * altitude) / camera_parameters.focal_length_mm


def get_fov_polygon(waypoint: Waypoint, fov_m: NDArray[np.float64]) -> Polygon:
    half_fov = fov_m / 2
    corners_relative = np.array(
        [
            [half_fov[0], half_fov[1]],
            [half_fov[0], -half_fov[1]],
            [-half_fov[0], -half_fov[1]],
            [-half_fov[0], half_fov[1]],
        ]
    )

    rotation_matrix = np.array(
        [
            [np.cos(-waypoint.heading), -np.sin(-waypoint.heading)],
            [np.sin(-waypoint.heading), np.cos(-waypoint.heading)],
        ]
    )
    rotated_corners = np.dot(corners_relative, rotation_matrix.T)

    corners_utm = rotated_corners + waypoint.location.utm_coordinate[:2]
    return Polygon([corners_utm[i, :] for i in range(corners_utm.shape[0])])


def georeference_detections(
    detections: list[Detection],
    waypoint: Waypoint,
    camera_parameters: CameraParameters,
) -> list[Location]:
    coordinates_image_px = np.array([d.coordinate for d in detections], dtype=np.uint16)

    fov_m = get_field_of_view_m(waypoint.location.gps_coordinate_lat_lon[2], camera_parameters)

    coordinates_local = (coordinates_image_px / camera_parameters.image_size - 0.5) * fov_m
    coordinates_local[:, 1] *= -1  # y-axis is inverted in UTM coordinates

    rotation_matrix = np.array(
        [
            [np.cos(-waypoint.heading), -np.sin(-waypoint.heading)],
            [np.sin(-waypoint.heading), np.cos(-waypoint.heading)],
        ]
    )
    coordinates_local_rotated = np.dot(coordinates_local, rotation_matrix.T)

    coordinates_utm = waypoint.location.utm_coordinate[:2] + coordinates_local_rotated

    return [Location.from_utm(coordinates_utm[i, :]) for i in range(coordinates_utm.shape[0])]
