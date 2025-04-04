from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from adaptive_planner import require_module, setup_logging
from adaptive_planner.location import Location
from adaptive_planner.utils import Annotation

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from Metashape import Camera, Chunk


log = setup_logging("metashape")


@require_module("Metashape")  # type: ignore[misc]
def load_project(project_file: Path, chunk_name: str | None = None) -> Chunk | None:
    import Metashape

    if not project_file.is_file() or project_file.suffix != ".psx":
        log.fatal(f"Agisoft Metashape project file '{project_file.name}' is not valid!")
        return None

    # Check if Agisoft Metashape is activated
    if not Metashape.app.activated:
        log.fatal("Agisoft Metashape is not activated. Activate Metashape first!")
        return None

    log.info("Agisoft Metashape is activated.")

    doc = Metashape.Document()
    doc.open(str(project_file))

    chunks = [c.label for c in doc.chunks]
    if chunk_name is not None:
        if chunk_name not in chunks:
            log.fatal(f"Chunk name {chunk_name} is not in project {project_file.name}!")
            return None
    else:
        if len(chunks) > 1:
            log.warning("Multiple chunk names available. Taking first one. Specify chunk name to control this...")

        chunk_name = chunks[0]

    chunk = doc.chunks[chunks.index(chunk_name)]

    if chunk.crs.authority != "EPSG::4326":
        log.fatal(f"Project CRS should be 'EPSG::4326' instead of '{chunk.crs.authority}'. Transform coordinate system in Agisoft first!")
        return None

    for camera in chunk.cameras:
        if not camera.transform:
            log.fatal(f"Camera {camera.label} does not have a transform. Align images first!")
            return None

    return chunk


@require_module("Metashape")  # type: ignore[misc]
def get_altitude_from_dem(chunk: Chunk, location: Location) -> float:
    import Metashape

    if chunk.elevation is None:
        raise RuntimeError("DEM is not generated in Metashape!")

    return chunk.elevation.altitude(Metashape.Vector([*location.gps_coordinate_lon_lat[:2]]))  # type: ignore[no-any-return]


@require_module("Metashape")  # type: ignore[misc]
def get_gsd_from_camera(chunk: Chunk, camera: Camera, sensor_width_mm: float, altitude: float | None = None) -> float:
    if altitude is None:
        altitude = get_altitude_from_dem(chunk, camera_to_location(camera))

    camera_height = camera.reference.location.z - altitude
    return (camera_height * sensor_width_mm) / (camera.sensor.width * camera.sensor.focal_length)  # type: ignore[no-any-return]


@require_module("Metashape")  # type: ignore[misc]
def get_camera_names(chunk: Chunk) -> dict[str, Path]:
    return {camera.label: Path(camera.photo.path) for camera in chunk.cameras}


@require_module("Metashape")  # type: ignore[misc]
def get_image_size_from_camera(camera: Camera) -> NDArray[np.uint16]:
    return np.array([camera.sensor.width, camera.sensor.height], dtype=np.uint16)


@require_module("Metashape")  # type: ignore[misc]
def get_camera_by_name(chunk: Chunk, camera_name: str) -> Camera | None:
    for camera in chunk.cameras:
        if camera.label == camera_name:
            return camera
    return None


@require_module("Metashape")  # type: ignore[misc]
def gps_to_pixel_coordinate(chunk: Chunk, camera: Camera, location: Location) -> NDArray[np.uint16] | None:
    import Metashape

    coordinate_lon_lat = location.gps_coordinate_lon_lat.copy()

    # If we don't have the altitude, retrieve it from the DEM
    if coordinate_lon_lat.shape[0] == 2:
        coordinate_lon_lat = np.array([*coordinate_lon_lat, get_altitude_from_dem(chunk, location)], dtype=np.float64)
    elif np.isnan(coordinate_lon_lat[2]):
        coordinate_lon_lat[2] = get_altitude_from_dem(chunk, location)

    coordinate_vector = Metashape.Vector(coordinate_lon_lat)  # Lon, Lat, Alt
    projected_location = camera.project(chunk.transform.matrix.inv().mulp(chunk.crs.unproject(coordinate_vector)))

    if projected_location is None:
        return None

    if (
        projected_location.x < 0
        or projected_location.x > camera.sensor.width
        or projected_location.y < 0
        or projected_location.y > camera.sensor.height
    ):
        return None

    return np.array([projected_location[0], projected_location[1]], dtype=np.uint16)


@require_module("Metashape")  # type: ignore[misc]
def pixel_to_gps_location(
    chunk: Chunk,
    camera: Camera,
    pixel_coordinates: NDArray[np.uint16] | Tensor | tuple[int, int],
    retry: bool = False,
) -> Location | None:
    import Metashape

    ray_origin = camera.unproject(Metashape.Vector([pixel_coordinates[0], pixel_coordinates[1], 0]))
    ray_target = camera.unproject(Metashape.Vector([pixel_coordinates[0], pixel_coordinates[1], 1]))
    internal_point = chunk.tie_points.pickPoint(ray_origin, ray_target)

    if internal_point is None:
        if retry:
            return None

        log.warning(
            f"Cannot pick point for {pixel_coordinates} for image {camera.label}, will try again with higher offset from image borders..."
        )
        pixel_coordinates = np.clip(pixel_coordinates, [100, 100], [camera.sensor.width - 100, camera.sensor.height - 100])
        return pixel_to_gps_location(chunk, camera, pixel_coordinates, retry=True)  # type: ignore[no-any-return]

    gps_location = chunk.crs.project(chunk.transform.matrix.mulp(internal_point))

    return Location(np.array([*gps_location], dtype=np.float64))


@require_module("Metashape")  # type: ignore[misc]
def distance_between_coordinates(
    chunk: Chunk, coordinate_lon_lat_1: NDArray[np.float64], coordinate_lon_lat_2: NDArray[np.float64]
) -> float:
    import Metashape

    coordinate_1 = Metashape.Vector(coordinate_lon_lat_1)
    coordinate_2 = Metashape.Vector(coordinate_lon_lat_2)

    # Calculate to internal coordinates (they are in meters)
    coordinate_1_internal = chunk.crs.unproject(coordinate_1)
    coordinate_2_internal = chunk.crs.unproject(coordinate_2)

    return np.linalg.norm(
        np.array([*coordinate_1_internal], dtype=np.float64) - np.array([*coordinate_2_internal], dtype=np.float64)
    ).item()


@require_module("Metashape")  # type: ignore[misc]
def get_visible_locations(chunk: Chunk, camera: Camera, all_locations: list[Location]) -> list[Location]:
    return [location for location in all_locations if gps_to_pixel_coordinate(chunk, camera, location) is not None]


@require_module("Metashape")  # type: ignore[misc]
def get_cameras_by_location(chunk: Chunk, location: Location) -> list[Camera]:
    cameras = []
    for camera in chunk.cameras:
        if gps_to_pixel_coordinate(chunk, camera, location) is not None:
            cameras.append(camera)
    return cameras


@require_module("Metashape")  # type: ignore[misc]
def get_annotations_by_camera(chunk: Chunk, camera: Camera, markers: list[Location], bbox_size: tuple[float, float]) -> list[Annotation]:
    annotations = []
    for marker in markers:
        image_coordinates = gps_to_pixel_coordinate(chunk, camera, marker)
        if image_coordinates is not None:
            bbox_size_px = np.array(bbox_size, dtype=np.float32) / camera.sensor.pixel_size
            bbox_size_px = bbox_size_px.astype(np.uint16)

            annotations.append(
                Annotation(
                    "",
                    max(0, image_coordinates[0] - bbox_size_px[0] // 2),
                    max(0, image_coordinates[1] - bbox_size_px[1] // 2),
                    bbox_size_px[0],
                    bbox_size_px[1],
                    marker.properties["class_name"],
                    camera.label,
                    np.array([camera.sensor.width, camera.sensor.height], dtype=np.uint16),
                )
            )

    return annotations


@require_module("Metashape")  # type: ignore[misc]
def camera_to_location(camera: Camera) -> Location:
    return Location(np.array([camera.reference.location.x, camera.reference.location.y]))
