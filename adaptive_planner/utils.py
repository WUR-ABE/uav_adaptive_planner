from __future__ import annotations

from binascii import hexlify, unhexlify
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from json import loads as load_json
from time import time
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.request import urlopen
from zlib import compress, decompress

import numpy as np
from numpy.typing import NDArray

from pyproj import CRS
from shapely import Polygon

from adaptive_planner.location import Location

if TYPE_CHECKING:
    from logging import Logger
    from types import TracebackType
    from typing import Any, Callable

    from numpy.typing import NDArray


class TileServers(Enum):
    OSM = auto()
    GOOGLE = auto()
    GOOGLE_SATELLITE = auto()
    ARCGIS = auto()


TILE_SERVER_URL = {
    TileServers.OSM: "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
    TileServers.GOOGLE: "https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga",
    TileServers.GOOGLE_SATELLITE: "https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga",
    TileServers.ARCGIS: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

# fmt: off
IMAGE_EXTENSIONS = [
    ".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif", ".bmp",
    ".raw", ".cr2", ".nef", ".arw", ".orf", ".dng", ".heif",
    ".heic", ".webp", ".svg", ".ico", ".psd", ".pdf", ".exr",
    ".jp2", ".3fr", ".x3f", ".pef", ".raf", ".k25",
]
# fmt: on


class LogProcessingTime:
    def __init__(self, logger: Logger, msg: str) -> None:
        self.log = logger
        self.msg = msg
        self.start_time: float | None = None

    def __enter__(self) -> LogProcessingTime:
        self.start_time = time()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        assert self.start_time is not None

        self.log.info(f"{self.msg} in {time() - self.start_time:.2f} seconds")


def img_to_str(image: NDArray[np.uint8]) -> str:
    header = f"{image.shape},{image.dtype}"
    combined_data = header.encode("utf-8") + b"||" + image.tobytes()
    return hexlify(compress(combined_data)).decode("utf-8")


def img_from_str(string: str) -> NDArray[np.uint8]:
    decompressed_data = decompress(unhexlify(string))
    header, array_data = decompressed_data.split(b"||", 1)
    shape, dtype = header.decode("utf-8").split(",")
    return np.frombuffer(array_data, dtype=dtype).reshape(eval(shape))


def polygon_to_crs(polygon: Polygon, crs: CRS) -> Polygon:
    return Polygon([Location(np.array(coord, dtype=np.float64)).to_crs(crs) for coord in polygon.exterior.coords])


def calculate_total_distance(flight_path: list[Location], use_3d: bool = True) -> float:
    distance = 0.0
    for i in range(len(flight_path) - 1):
        distance += flight_path[i].get_distance(flight_path[i + 1], use_3d=use_3d)

    return distance


def msl_from_server(location: Location) -> float:
    endpoint = (
        f"https://api.opentopodata.org/v1/eudem25m?locations={location.gps_coordinate_lat_lon[0]},{location.gps_coordinate_lat_lon[1]}"
    )

    with urlopen(endpoint) as response:
        data = response.read().decode("utf-8")

    parsed_data = load_json(data)
    if parsed_data.get("status") == "OK":
        return float(parsed_data["results"][0]["elevation"])

    raise ConnectionError("Could not connect to opentopodata!")


def create_distance_matrix(locations_1: list[Location], locations_2: list[Location], distance_threshold: float) -> NDArray[np.float32]:
    distance_matrix = np.zeros((len(locations_1), len(locations_2)), dtype=np.float32)
    for i, loc_1 in enumerate(locations_1):
        for j, loc_2 in enumerate(locations_2):
            distance = loc_1.get_distance(loc_2)
            distance_matrix[i, j] = distance if distance <= distance_threshold else 1e7  # just really high value
    return distance_matrix


KeyType = TypeVar("KeyType")
InputType1 = TypeVar("InputType1")
InputType2 = TypeVar("InputType2")
ResultsType = TypeVar("ResultsType")


def _fn_wrapper(data: tuple[Callable[..., ResultsType], list[Any], dict[str, Any]]) -> ResultsType:
    fn, args, kwargs = data
    return fn(*args, **kwargs)


def parallel_execute(
    input_data: dict[KeyType, InputType1],
    fn: Callable[[InputType1], ResultsType],
    **kwargs: Any,
) -> dict[KeyType, ResultsType]:
    output_dict = {}

    with ProcessPoolExecutor() as executor:
        for key, result in zip(input_data.keys(), executor.map(_fn_wrapper, [(fn, [arg], kwargs) for arg in input_data.values()])):
            output_dict[key] = result

    return output_dict


def parallel_execute2(
    input_data: dict[KeyType, tuple[InputType1, InputType2]],
    fn: Callable[[InputType1, InputType2], ResultsType],
    **kwargs: Any,
) -> dict[KeyType, ResultsType]:
    output_dict = {}

    with ProcessPoolExecutor() as executor:
        for key, result in zip(input_data.keys(), executor.map(_fn_wrapper, [(fn, args, kwargs) for args in input_data.values()])):
            output_dict[key] = result

    return output_dict


def get_location_by_name(locations: list[Location], name: str) -> Location:
    for loc in locations:
        if loc.properties["name"] == name:
            return loc

    raise NameError(f"Could not find '{name}'!")


@dataclass
class Waypoint:
    location: Location
    heading: float


@dataclass
class CameraParameters:
    image_size: tuple[int, int]
    sensor_size_mm: tuple[float, float]
    focal_length_mm: float


@dataclass
class ImageMetaData:
    name: str
    timestamp: datetime
    gps_coordinate: NDArray[np.float64]
    flight_rpy: NDArray[np.float32]
    gimbal_rpy: NDArray[np.float32]
    image_size: NDArray[np.int16]
    sensor_size: NDArray[np.float32]
    field_of_view: float
    focal_length: float

    def to_camera_parameters(self) -> CameraParameters:
        return CameraParameters(
            self.image_size.tolist(),
            self.sensor_size.tolist(),
            self.focal_length,
        )

    @classmethod
    def from_exif(cls, exif_data: dict[str, Any], image_name: str) -> ImageMetaData:
        timestamp = datetime.strptime(exif_data["EXIF:DateTimeOriginal"], r"%Y:%m:%d %H:%M:%S")

        gps_coordinate = [
            exif_data["EXIF:GPSLatitude"],
            exif_data["EXIF:GPSLongitude"],
            float(exif_data["EXIF:GPSAltitude"]),  # WGS84
        ]
        flight_rpy = [
            exif_data["XMP:FlightRollDegree"],
            exif_data["XMP:FlightPitchDegree"],
            exif_data["XMP:FlightYawDegree"],
        ]
        gimbal_rpy = [
            exif_data["XMP:GimbalRollDegree"],
            exif_data["XMP:GimbalPitchDegree"],
            exif_data["XMP:GimbalYawDegree"],
        ]

        match (exif_data["EXIF:Make"], exif_data["XMP:Model"]):
            case ("DJI", "ZenmuseP1"):
                sensor_size = (35.9, 24.0)
            case _:
                raise NotImplementedError(f"Sensor size of {exif_data['EXIF:Make']} {exif_data['XMP:Model']} camera not implemented!")

        return cls(
            image_name,
            timestamp,
            np.array(gps_coordinate, dtype=np.float64),
            np.array(flight_rpy, dtype=np.float32),
            np.array(gimbal_rpy, dtype=np.float32),
            np.array([exif_data["File:ImageWidth"], exif_data["File:ImageHeight"]], dtype=np.float32),
            np.array(sensor_size, dtype=np.float32),
            exif_data["Composite:FOV"],
            float(exif_data["EXIF:FocalLength"]),
        )


@dataclass
class Annotation:
    name: str
    x: int  # Top-left
    y: int  # Top-left
    w: int
    h: int
    class_name: str
    image_name: str
    image_size: NDArray[np.uint16]
    tags: list[str] = field(default_factory=list)

    @property
    def coordinate(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (
            *self.coordinate,
            self.w,
            self.h,
        )

    @property
    def yolo_xywh(self) -> tuple[float, float, float, float]:
        return (
            self.coordinate[0] / self.image_size[0],
            self.coordinate[1] / self.image_size[1],
            self.w / self.image_size[0],
            self.h / self.image_size[1],
        )

    @property
    def x1y1x2y2(self) -> tuple[float, float, float, float]:
        return (
            self.x,
            self.y,
            self.x + self.w,
            self.y + self.h,
        )
