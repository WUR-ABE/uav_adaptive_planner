from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, cast
from yaml import safe_load

import cv2
import numpy as np

from pyproj import CRS
from rasterio.plot import reshape_as_image
from rasterio.transform import Affine, rowcol
from rasterio.warp import Resampling, reproject
from scipy.spatial.transform import Rotation as R
from shapely import Polygon
from shapely.ops import unary_union

from adaptive_planner import setup_logging
from adaptive_planner.executor import Executor
from adaptive_planner.georeference import get_field_of_view_m, get_orthomosaic_resolution
from adaptive_planner.io.kml import read_kml_file
from adaptive_planner.io.orthomosaic import extract_roi_from_orthomosaic, get_tile_polygons, get_tiles_to_load, load_tiles
from adaptive_planner.io.topcon import read_topcon_data
from adaptive_planner.location import WGS_84, Location
from adaptive_planner.utils import CameraParameters, LogProcessingTime, Waypoint

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from rasterio import DatasetReader

    from adaptive_planner.field import Field

log = setup_logging(__name__)


class OrthomosaicSimException(RuntimeError):
    pass


class OrthomosaicRotationMethod(Enum):
    FAST = auto()
    BEST = auto()


@dataclass
class OrthomosaicSimConfig:
    rotation_method: OrthomosaicRotationMethod = OrthomosaicRotationMethod.FAST
    seed: int = -1

    # Uncertainty parameters
    position_uncertainty_m: float = 0.0
    altitude_uncertainty_m: float = 0.0
    heading_uncertainty_deg: float = 0.0
    roll_uncertainty_deg: float = 0.0
    pitch_uncertainty_deg: float = 0.0

    # Number of objects parameters
    override_number_of_objects: int | None = None
    override_locations_file: str | None = None
    gt_locations_file: str | None = None
    hide_object_color: tuple[int, int, int] = (181, 179, 158)  # RGB
    hide_object_size: float = 0.60  # meter
    add_object_size: float = 0.10  # meter

    gt_locations: list[Location] | None = field(default=None, repr=False, init=False)
    override_locations: list[Location] | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self.rotation_method = OrthomosaicRotationMethod(self.rotation_method)

        if self.override_number_of_objects is not None:
            if self.gt_locations_file is None or self.override_locations_file is None:
                raise OrthomosaicSimException(
                    "When changing the number of objects, the 'gt_locations_file' and 'override_locations_file' parameters should be specified!"
                )

            self.gt_locations = read_topcon_data(Path(self.gt_locations_file))
            self.override_locations = read_kml_file(Path(self.override_locations_file))

    def get_override_location_names(self) -> list[str]:
        assert self.override_locations is not None
        return [loc.properties["name"] for loc in self.override_locations]


@dataclass
class AddedLocation:
    location: Location
    image: NDArray[np.uint8]
    rotation: float


class OrthomosaicSim(Executor):
    def __init__(self, field: Field, output_folder: Path | None = None, config: OrthomosaicSimConfig = OrthomosaicSimConfig()) -> None:
        self.field = field
        self.output_folder = output_folder
        self.config = config

        self.rng = np.random.default_rng(seed=None if self.config.seed == -1 else self.config.seed)

        assert self.field.orthomosiac_scheme_file is not None, "Orthomosaic scheme file should be defined!"

        self.tiles = get_tile_polygons(self.field.orthomosiac_scheme_file)

        self._loaded_raster: DatasetReader | None = None
        self._loaded_tiles: set[Path] | None = None
        self._loaded_polygon: Polygon | None = None

        self._data_transform: Affine | None = None
        self._data_buffer: NDArray[np.uint8] | None = None

        self.object_locations_to_add: list[AddedLocation] = []
        self.object_locations_to_remove: list[Location] = []

        if self.config.override_number_of_objects is not None:
            assert self.config.override_locations is not None
            assert self.config.gt_locations is not None

            if len(self.config.override_locations) < self.config.override_number_of_objects:
                raise OrthomosaicSimException(
                    f"There are too less override locations ({len(self.config.override_locations)}) "
                    f"for the requested number of override locations ({self.config.override_number_of_objects})!"
                )

            object_names = self.config.get_override_location_names()[: self.config.override_number_of_objects]
            log.info(f"Object names in the orthomosaic: {object_names}")

            if len(self.config.gt_locations) > self.config.override_number_of_objects:
                for gt in self.config.gt_locations:
                    if gt.properties["name"] in object_names:
                        continue

                    self.object_locations_to_remove.append(gt)

                log.info(f"Removing {len(self.object_locations_to_remove)} objects from the orthomosiac")

            if len(self.config.gt_locations) < self.config.override_number_of_objects:
                object_names = object_names[len(self.config.gt_locations) :]

                for loc in self.config.override_locations:
                    if loc.properties["name"] not in object_names:
                        continue

                    self.object_locations_to_add.append(
                        AddedLocation(loc, loc.properties["object_image"], loc.properties["object_image_angle"])
                    )

                log.info(f"Adding {len(self.object_locations_to_add)} objects to the orthomosiac")

    @classmethod
    def from_file(cls: type[OrthomosaicSim], field: Field, output_folder: Path | None, config_file: Path) -> OrthomosaicSim:
        with config_file.open("r") as file_handler:
            config = OrthomosaicSimConfig(**safe_load(file_handler))

        return cls(field, output_folder, config=config)

    @property
    def camera_parameters(self) -> CameraParameters:
        """
        returns: Camera parameters corresponding to DJI Zenmuse P1-camera.
        """
        return CameraParameters(
            (8192, 5460),
            (35.9, 24.0),
            35.0,
        )

    def get_image_at_coordinate(self, waypoint: Waypoint, **kwargs: Any) -> NDArray[np.uint8]:
        waypoint = self.get_uncertain_waypoint(waypoint)

        tiles_to_load = get_tiles_to_load(self.tiles, waypoint, self.camera_parameters)
        if len(tiles_to_load) == 0:
            raise RuntimeError(f"GPS coordinate {waypoint.location.gps_coordinate_lon_lat[:2].round(6)} is not in any tile!")

        if self._loaded_tiles is None or not tiles_to_load.issubset(self._loaded_tiles):
            self._loaded_raster = load_tiles(list(tiles_to_load), parallel=True)
            self._loaded_tiles = tiles_to_load
            self._loaded_polygon = cast(Polygon, unary_union([self.tiles[t] for t in tiles_to_load]))

            self._data_heading = None
            self._data_buffer = None

            if self._loaded_raster.crs.linear_units != "metre":
                raise RuntimeError(
                    f"The loaded raster linear units '{self._loaded_raster.crs.linear_units}' are not in meters! Use an"
                    " orthomosaic with UTM coordinates."
                )

        assert self._loaded_raster is not None
        assert self._loaded_tiles is not None
        assert self._loaded_polygon is not None

        transform, array_size = (
            self.calculate_raster_transform_best(waypoint)
            if self.config.rotation_method == OrthomosaicRotationMethod.BEST
            else self.calculate_raster_transform_fast(waypoint)
        )

        if self._data_transform is None or not self._data_transform.almost_equals(transform):
            self._data_buffer = None
            self._data_transform = None

        if self._data_buffer is None or self._data_transform is None:
            with LogProcessingTime(log, "Reprojected orthomosaic"):
                self._data_transform = transform
                data = np.zeros(array_size, dtype=self._loaded_raster.dtypes[0])

                reproject(
                    self._loaded_raster.read(),
                    data,
                    src_transform=self._loaded_raster.transform,
                    src_crs=self._loaded_raster.crs,
                    dst_transform=self._data_transform,
                    dst_crs=self._loaded_raster.crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )

                self._data_buffer = np.ascontiguousarray(reshape_as_image(data))

            if len(self.object_locations_to_add) > 0:
                with LogProcessingTime(log, f"Added {len(self.object_locations_to_add)} objects"):
                    self.add_objects_to_raster(self._data_buffer, self._data_transform)

            if len(self.object_locations_to_remove) > 0:
                with LogProcessingTime(log, f"Removed {len(self.object_locations_to_remove)} objects"):
                    self.remove_objects_from_raster(self._data_buffer, self._data_transform)

        coordinate = self.transform_to_raster_coordinates(waypoint.location)
        drone_pixel_coordinates = rowcol(self._data_transform, *coordinate[:2])
        fov_m = get_field_of_view_m(coordinate[2], self.camera_parameters)
        fov_px = (fov_m / get_orthomosaic_resolution(self._data_transform)).astype(np.uint16)

        img_data = extract_roi_from_orthomosaic(self._data_buffer, drone_pixel_coordinates, np.flip(fov_px), pad_value=255)

        return cv2.resize(img_data, self.camera_parameters.image_size, interpolation=cv2.INTER_NEAREST)  # type: ignore[return-value]

    def enable(self) -> None:
        return

    def finish(self) -> None:
        return

    def get_uncertain_waypoint(self, waypoint: Waypoint) -> Waypoint:
        """
        Function to apply uncertainty on the GPS position of the drone and on the drone's heading.

        :param waypoint: Waypoint of the drone
        :returns: Uncertain waypoint of the drone with heading in radians.
        """
        position_uncertainty = self.rng.uniform(-self.config.position_uncertainty_m, self.config.position_uncertainty_m, size=2)
        altitude_uncertainty = self.rng.uniform(-self.config.altitude_uncertainty_m, self.config.altitude_uncertainty_m)
        heading_uncertainty_deg = self.rng.uniform(-self.config.heading_uncertainty_deg, self.config.heading_uncertainty_deg)
        roll_uncertainty_deg = self.rng.uniform(-self.config.roll_uncertainty_deg, self.config.roll_uncertainty_deg)
        pitch_uncertainty_deg = self.rng.uniform(-self.config.pitch_uncertainty_deg, self.config.pitch_uncertainty_deg)

        # Calculate orientation error
        rotation_matrix = R.from_euler(
            "xyz", [roll_uncertainty_deg, pitch_uncertainty_deg, heading_uncertainty_deg], degrees=True
        ).as_matrix()
        position_uncertainty_orientation = rotation_matrix @ np.array([0, 0, waypoint.location.altitude + altitude_uncertainty])

        # Apply position error
        location = waypoint.location.copy()
        utm_coordinates = location.utm_coordinate[:2] + position_uncertainty + position_uncertainty_orientation[:2]
        uncertain_location = Location.from_crs(
            np.array([*utm_coordinates, location.altitude + altitude_uncertainty], dtype=np.float64),
            Location.get_utm_crs(location.gps_coordinate_lon_lat),
        )

        log.info(
            f"Applied uncertainty: location={location.utm_coordinate - uncertain_location.utm_coordinate}, heading={heading_uncertainty_deg}"
        )
        return Waypoint(uncertain_location, waypoint.heading + np.deg2rad(heading_uncertainty_deg))

    def calculate_raster_transform_best(self, waypoint: Waypoint) -> tuple[Affine, tuple[int, int, int]]:
        assert self._loaded_raster is not None

        coordinate = self.transform_to_raster_coordinates(waypoint.location)
        drone_pixel_coordinates = rowcol(self._loaded_raster.transform, *coordinate[:2])
        rotated_transform = self._loaded_raster.transform * Affine.rotation(
            np.rad2deg(waypoint.heading), pivot=np.flip(drone_pixel_coordinates)
        )

        width, height = self._loaded_raster.width, self._loaded_raster.height
        corners = [
            rotated_transform * (0, 0),
            rotated_transform * (width, 0),
            rotated_transform * (0, height),
            rotated_transform * (width, height),
        ]
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)

        new_width = int((max_x - min_x) / abs(rotated_transform.a))
        new_height = int((max_y - min_y) / abs(rotated_transform.e))

        offset_transform = rotated_transform * Affine.translation(
            (min_x - self._loaded_raster.transform.c), (min_y - self._loaded_raster.transform.f)
        )

        return offset_transform, (self._loaded_raster.count, new_height, new_width)

    def calculate_raster_transform_fast(self, waypoint: Waypoint) -> tuple[Affine, tuple[int, int, int]]:
        assert self._loaded_raster is not None

        coordinate = self.transform_to_raster_coordinates(waypoint.location)
        drone_pixel_coordinates = rowcol(self._loaded_raster.transform, *coordinate[:2])
        rotated_transform = self._loaded_raster.transform * Affine.rotation(
            np.rad2deg(waypoint.heading), pivot=np.flip(drone_pixel_coordinates)
        )

        return rotated_transform, (self._loaded_raster.count, self._loaded_raster.height, self._loaded_raster.width)

    def add_objects_to_raster(self, raster: NDArray[np.uint8], raster_transform: Affine) -> None:
        assert self._loaded_polygon is not None

        for object_to_add in self.object_locations_to_add:
            if not self._loaded_polygon.contains(object_to_add.location.to_point()):
                continue

            object_raster_coordinates = self.transform_to_raster_coordinates(object_to_add.location)
            object_pixel_coordinates = rowcol(raster_transform, *object_raster_coordinates[:2])
            object_pixel_coordinates = np.array(object_pixel_coordinates).astype(int)

            object_size = self.config.add_object_size / get_orthomosaic_resolution(raster_transform).mean()
            object_image = self.resize_with_aspect_ratio(object_to_add.image, min_width=object_size, min_height=object_size)
            object_image = self.rotate_image(object_image, object_to_add.rotation)

            # Skip objects that are rotated out of the image
            object_image_shape = np.array(object_image.shape[:2], dtype=int)
            if np.any(object_pixel_coordinates + object_image_shape // 2 <= [0, 0]) or np.any(
                object_pixel_coordinates - object_image_shape // 2 >= raster.shape[:2]
            ):
                log.info(
                    f"Skip adding object '{object_to_add.location.properties['name']}' because it is completely rotated out of the image"
                )
                continue

            self.paste_image(raster, object_image, object_pixel_coordinates)

    def remove_objects_from_raster(self, raster: NDArray[np.uint8], raster_transform: Affine) -> None:
        assert self._loaded_polygon is not None

        for object_location in self.object_locations_to_remove:
            if not self._loaded_polygon.contains(object_location.to_point()):
                continue

            mask_size_px = self.config.hide_object_size / get_orthomosaic_resolution(raster_transform)
            object_raster_coordinates = self.transform_to_raster_coordinates(object_location)
            object_pixel_coordinates = rowcol(raster_transform, *object_raster_coordinates[:2])

            # Skip objects that are rotated out of the image
            if np.any(object_pixel_coordinates + mask_size_px // 2 <= [0, 0]) or np.any(
                object_pixel_coordinates - mask_size_px // 2 >= raster.shape[:2]
            ):
                log.info(f"Skip removing object '{object_location.properties['name']}' because it is completely rotated out of the image")
                continue

            cv2.rectangle(
                raster,
                np.flip(np.array(object_pixel_coordinates - mask_size_px // 2).astype(int)).tolist(),
                np.flip(np.array(object_pixel_coordinates + mask_size_px // 2).astype(int)).tolist(),
                self.config.hide_object_color,
                thickness=-1,
            )

    def transform_to_raster_coordinates(self, location: Location) -> NDArray[np.float64]:
        assert self._loaded_raster is not None

        coordinate = location.gps_coordinate_lon_lat.copy()
        if self._loaded_raster.crs.to_epsg() != WGS_84.to_epsg():
            crs = CRS.from_epsg(self._loaded_raster.crs.to_epsg())
            coordinate[:2] = location.to_crs(crs)[:2]

        return coordinate

    @staticmethod
    def resize_with_aspect_ratio(
        image: NDArray[np.uint8], min_width: int | None = None, min_height: int | None = None
    ) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        if min_width is None and min_height is None:
            return image

        ratio = max((min_width / w if min_width else 0), (min_height / h if min_height else 0))

        return cv2.resize(image, (int(w * ratio), int(h * ratio)))  # type: ignore[return-value]

    @staticmethod
    def rotate_image(image: NDArray[np.uint8], angle: float) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), -np.rad2deg(angle), 1.0)
        cos_angle = np.abs(rotation_matrix[0, 0])
        sin_angle = np.abs(rotation_matrix[0, 1])

        nw = int((h * sin_angle) + (w * cos_angle))
        nh = int((h * cos_angle) + (w * sin_angle))

        rotation_matrix[0, 2] += (nw / 2) - cx
        rotation_matrix[1, 2] += (nh / 2) - cy

        return cv2.warpAffine(image, rotation_matrix, (nw, nh))  # type: ignore[return-value]

    @staticmethod
    def paste_image(
        background_img: NDArray[np.uint8],
        foreground_img: NDArray[np.uint8],
        foreground_coordinates: NDArray[np.int_] | tuple[int, int],
    ) -> None:
        foreground_size = np.array(foreground_img.shape[:2], dtype=int)
        background_size = np.array(background_img.shape[:2], dtype=int)

        top_left_coordinates = foreground_coordinates - foreground_size // 2

        top_left = np.maximum(0, -top_left_coordinates)
        bottom_right = np.minimum(foreground_size, foreground_size - (top_left_coordinates + foreground_size - background_size))
        image = foreground_img[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

        top_left = np.maximum(0, top_left_coordinates)
        bottom_right = np.minimum(background_size, top_left_coordinates + foreground_size)

        # Remove black pixels
        mask = cv2.inRange(image, (5, 5, 5), (255, 255, 255))  # type: ignore[call-overload]
        non_black_pixels = np.nonzero(mask)
        background_img[
            top_left[0] : top_left[0] + image.shape[0],
            top_left[1] : top_left[1] + image.shape[1],
        ][non_black_pixels] = image[non_black_pixels]


if __name__ == "__main__":
    from pathlib import Path
    import tracemalloc

    from cv2 import imshow, resize, waitKey

    from ultralytics import YOLO

    from adaptive_planner.field import Field

    tracemalloc.start()

    yolo = YOLO("adaptive_planner/best_n.pt")

    STEP_SIZE = 2.0  # m
    ALTITUDE_STEP_SIZE = 2.0  # m
    HEADING_STEP_SIZE = np.deg2rad(15.0)  # rad

    fly_field = Field.from_file(Path("fields/clustered_1.yaml"))
    sim = OrthomosaicSim.from_file(fly_field, None, Path("experiments/number_of_objects/executor_200_objects_clustered_1.yaml"))

    # Define start position and heading
    coordinates = Location(np.array([5.66793408413133, 51.99141508503395, 12.0], dtype=np.float64))
    utm_coordinates = coordinates.utm_coordinate

    # utm_coordinates = np.array([683147.9878289072, 5763379.117088596, 12.0])
    utm_crs = CRS.from_string("EPSG:32631")
    heading = 1.3089969388415839

    # utm_coordinates = np.array([*Transformer.from_crs(WGS_84, utm_crs).transform(51.99110034459039, 5.66779602939503), 20.0])
    # print(utm_coordinates)
    np.set_printoptions(precision=15)
    while True:
        print(f"Drone UTM coordinates: {utm_coordinates}")
        print(f"Drone heading: {heading}")

        img = sim.get_image_at_coordinate(Waypoint(Location.from_utm(utm_coordinates), heading))
        # cv2.imwrite("test.png", img[:, :, ::-1])
        # break

        prediction = yolo.predict(img[:, :, ::-1])
        imshow("Drone image", resize(prediction[0].plot(), (1366, 768)))  # RGB -> BGR

        key = waitKey(0) & 0xFF

        # Since coordinates are in UTM, we can simply add or subtract distances from the
        # easting and northing coordinates. Rotation matrices are counter-clockwise.
        rotation_matrix = np.array(
            [
                [np.cos(-heading), -np.sin(-heading)],
                [np.sin(-heading), np.cos(-heading)],
            ],
            dtype=np.float32,
        )

        if key == 27:
            break
        elif key == ord("w"):
            # Fly north
            utm_coordinates[:2] += np.dot([0, STEP_SIZE], rotation_matrix.T)
        elif key == ord("a"):
            # Fly west
            utm_coordinates[:2] += np.dot([-STEP_SIZE, 0], rotation_matrix.T)
        elif key == ord("s"):
            # Fly south
            utm_coordinates[:2] += np.dot([0, -STEP_SIZE], rotation_matrix.T)
        elif key == ord("d"):
            # Fly east
            utm_coordinates[:2] += np.dot([STEP_SIZE, 0], rotation_matrix.T)
        elif key == ord(","):
            utm_coordinates[2] += ALTITUDE_STEP_SIZE
        elif key == ord("."):
            utm_coordinates[2] -= ALTITUDE_STEP_SIZE
            utm_coordinates[2] = max(utm_coordinates[2], 2.0)
        elif key == ord("q"):
            # Rotate counterclockwise
            heading -= HEADING_STEP_SIZE
        elif key == ord("e"):
            # Rotate clockwise
            heading += HEADING_STEP_SIZE

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    def get_variable_info() -> None:
        for stat in top_stats[:20]:
            print(f"File: {stat.traceback[0].filename}, Line: {stat.traceback[0].lineno}, Size: {stat.size / (1024**2):.2f} MB")

    get_variable_info()

    tracemalloc.stop()
