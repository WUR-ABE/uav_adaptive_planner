from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pyproj import CRS, Transformer
from shapely import Point

if TYPE_CHECKING:
    from numpy.typing import NDArray

WGS_84 = CRS.from_epsg(4326)  # WGS84 CRS


def switch_lon_lat(coordinate: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([coordinate[1], coordinate[0], *coordinate[2:]])


@dataclass
class Location:
    gps_coordinate_lon_lat: NDArray[np.float64]

    properties: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> Location:
        return deepcopy(self)

    @property
    def gps_coordinate_lat_lon(self) -> NDArray[np.float64]:
        return switch_lon_lat(self.gps_coordinate_lon_lat)

    @property
    def utm_coordinate(self) -> NDArray[np.float64]:
        transformer = Transformer.from_crs(WGS_84, self.get_utm_crs(self.gps_coordinate_lon_lat))
        return np.array([*transformer.transform(*self.gps_coordinate_lat_lon[:2]), *self.gps_coordinate_lon_lat[2:]], dtype=np.float64)

    @property
    def has_altitude(self) -> bool:
        return self.gps_coordinate_lat_lon.shape[0] == 3

    @property
    def altitude(self) -> float:
        assert self.has_altitude
        return self.gps_coordinate_lat_lon[2]  # type: ignore[no-any-return]

    def to_point(self) -> Point:
        return Point(*self.utm_coordinate[:2])

    def __str__(self) -> str:
        return (
            f"({self.gps_coordinate_lon_lat[0]}, {self.gps_coordinate_lon_lat[1]}, {self.gps_coordinate_lon_lat[2]})"
            if self.has_altitude
            else f"({self.gps_coordinate_lon_lat[0]}, {self.gps_coordinate_lon_lat[1]})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Location):
            return False
        return np.array_equal(self.gps_coordinate_lon_lat, other.gps_coordinate_lon_lat) and self.properties == other.properties

    def get_distance(self, other: Location, use_3d: bool = False) -> float:
        if use_3d and self.has_altitude and other.has_altitude:
            return float(np.linalg.norm(self.utm_coordinate - other.utm_coordinate))
        return float(np.linalg.norm(self.utm_coordinate[:2] - other.utm_coordinate[:2]))

    def get_heading(self, other: Location) -> float:
        coordinate_1_utm = self.utm_coordinate[:2]
        coordinate_2_utm = other.utm_coordinate[:2]
        return np.arctan2(coordinate_2_utm[0] - coordinate_1_utm[0], coordinate_2_utm[1] - coordinate_1_utm[1]) % (2 * np.pi)  # type: ignore[no-any-return]

    def to_crs(self, crs: CRS) -> NDArray[np.float64]:
        transformer = Transformer.from_crs(WGS_84, crs)

        assert crs.coordinate_system is not None
        if len(crs.coordinate_system.to_json_dict()["axis"]) > 2:
            return np.array(transformer.transform(*self.gps_coordinate_lat_lon), dtype=np.float64)

        return np.array([*transformer.transform(*self.gps_coordinate_lat_lon[:2]), *self.gps_coordinate_lat_lon[2:]], dtype=np.float64)

    @classmethod
    def from_point(cls: type[Location], point: Point) -> Location:
        return cls.from_utm(np.array((point.x, point.y), dtype=np.float64))

    @classmethod
    def from_crs(cls, coordinate: NDArray[np.float64], crs: CRS) -> Location:
        coordinate = coordinate.copy()

        # Only pass altitude when CRS is 3D, else just copy the altitude
        assert crs.coordinate_system is not None
        if len(crs.coordinate_system.to_json_dict()["axis"]) > 2:
            transformer = Transformer.from_crs(crs, WGS_84.to_3d())
            return cls(np.array(switch_lon_lat(np.array(transformer.transform(*coordinate), dtype=np.float64)), dtype=np.float64))

        transformer = Transformer.from_crs(crs, WGS_84)
        coordinate[:2] = switch_lon_lat(np.array(transformer.transform(*coordinate[:2]), dtype=np.float64))
        return cls(coordinate)

    @classmethod
    def from_utm(cls, utm_coordinate: NDArray[np.float64], utm_zone: str = "31") -> Location:
        return Location.from_crs(utm_coordinate, CRS(f"+proj=utm +zone={utm_zone} +ellps=WGS84"))

    @staticmethod
    def get_utm_crs(gps_lon_lat: NDArray[np.float64]) -> CRS:
        utm_epsg = round((gps_lon_lat[0] + 180) / 6) + 32600
        return CRS.from_string(f"EPSG:{utm_epsg}")
