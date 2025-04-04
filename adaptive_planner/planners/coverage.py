from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import fields2cover as f2c
import numpy as np

from adaptive_planner import setup_logging
from adaptive_planner.georeference import get_field_of_view_m
from adaptive_planner.location import Location
from adaptive_planner.planners import Planner
from adaptive_planner.utils import Waypoint

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from adaptive_planner.field import Field
    from adaptive_planner.utils import CameraParameters

log = setup_logging(__name__)


@dataclass
class CoveragePlannerConfig:
    objective: f2c.OBJ_Base_SG_OBJ = f2c.OBJ_NSwathModified()
    sorter: f2c.RP_Single_cell_order_base_class = f2c.RP_Boustrophedon()

    sideways_overlap: float = 0.1
    forward_overlap: float = 0.1

    def as_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["objective"] = type(self.objective).__name__
        d["sorter"] = type(self.sorter).__name__
        return d


class CoveragePlanner(Planner):
    config: CoveragePlannerConfig

    def __init__(self, field: Field, camera_parameters: CameraParameters, config: CoveragePlannerConfig) -> None:
        self.field = field
        self.camera_parameters = camera_parameters
        self.config = config

    @property
    def config_dict(self) -> dict[str, Any]:
        return self.config.as_dict()

    def plan(self, start_position: Location, altitude: float, output_file: Path | None = None) -> list[Waypoint]:
        self.parse_field()
        start_point = f2c.Point(*start_position.gps_coordinate_lon_lat[:2])

        robot = self.create_robot(altitude, self.camera_parameters, self.config.sideways_overlap)

        # Convert to UTM
        f2c.Transform.transformToUTM(self._field)

        # Generate constant headland
        const_hl = f2c.HG_Const_gen()
        no_hl = const_hl.generateHeadlands(self._field.getField(), 0)

        # Generate swaths
        bf = f2c.SG_BruteForce()
        bf.setAllowOverlap(True)
        swaths = bf.generateBestSwaths(self.config.objective, robot.getCovWidth(), no_hl.getGeometry(0))

        # Sort swaths
        swaths = self.config.sorter.genSortedSwaths(swaths)

        # Custom ordering the swaths so that the first swath is close to the start point of the drone. Therefore,
        # transfer from the local coordinate system to GPS, to calculate the distance with the start point. Based
        # on the distance, reverse the swaths when needed.
        _swaths_tmp = f2c.Transform.transformToPrevCRS(swaths, self._field)
        distances = [
            self.get_distance(start_point, _swaths_tmp[0].startPoint(), is_gps=True),
            self.get_distance(start_point, _swaths_tmp[0].endPoint(), is_gps=True),
            self.get_distance(start_point, _swaths_tmp[_swaths_tmp.size() - 1].startPoint(), is_gps=True),
            self.get_distance(start_point, _swaths_tmp[_swaths_tmp.size() - 1].endPoint(), is_gps=True),
        ]

        if np.argmin(distances) in (1, 3):
            for swath in swaths:
                swath.reverse()

        if np.argmin(distances) in (2, 3):
            swaths.reverse()

        path_planner = f2c.PP_PathPlanning()
        dubins = f2c.PP_DubinsCurves()
        path = path_planner.planPath(robot, swaths, dubins)

        fov = get_field_of_view_m(altitude, self.camera_parameters)
        path = path.discretizeSwath(fov[1] * (1 - self.config.forward_overlap))

        # Transform back to GPS again
        path_gps = f2c.Transform.transformToPrevCRS(path, self._field)

        # Extract waypoints
        base_fligth_path: list[Waypoint] = []
        on_swath = False
        for state in path_gps.getStates():
            if state.type != f2c.PathSectionType_TURN or on_swath:
                location = Location(np.array([state.point.getX(), state.point.getY(), altitude], dtype=np.float64))
                heading = base_fligth_path[-1].location.get_heading(location) if len(base_fligth_path) else 0.0
                base_fligth_path.append(Waypoint(location, heading))
                on_swath = state.type != f2c.PathSectionType_TURN

        # Update heading of first waypoint to point to the next one
        if len(base_fligth_path) > 1:
            base_fligth_path[0].heading = base_fligth_path[1].heading

        if output_file is not None:
            f2c.Visualizer.figure()
            f2c.Visualizer.plot(self._field)
            f2c.Visualizer.plot(swaths)
            f2c.Visualizer.save(str(output_file))

        return base_fligth_path

    def set_objects_and_end_location(self, locations: list[Location], end_location: Location | None) -> None:
        raise NotImplementedError

    def parse_field(self) -> None:
        fields = f2c.Fields()
        f2c.Parser.importJson(str(self.field.boundary_file), fields)
        self._field = fields[0]
        self._field.setEPSGCoordSystem(4326)  # Field file should be in WGS84

    def create_field(self, polygon: NDArray[np.float64]) -> None:
        ring = f2c.LinearRing()
        for p in polygon:
            ring.addPoint(f2c.Point(*p))

        # Repeat first point to get closed LineString
        ring.addPoint(f2c.Point(*polygon[0, :]))

        cell = f2c.Cell()
        cell.addRing(ring)

        cells = f2c.Cells()
        cells.addGeometry(cell)

        self._field = f2c.Field(cells)
        self._field.setEPSGCoordSystem(4326)

    def get_start_location_from_field(self, altitude: float) -> Location:
        location = CoveragePlanner.f2c_point_to_location(self._field.getRefPoint())
        location.gps_coordinate_lon_lat[2] = altitude
        return location

    @staticmethod
    def create_robot(altitude: float, camera_parameters: CameraParameters, overlap: float) -> f2c.Robot:
        fov = get_field_of_view_m(altitude, camera_parameters)

        robot = f2c.Robot(1.0, fov[0] * (1 - overlap))
        robot.setMinTurningRadius(0)
        return robot

    @staticmethod
    def get_distance(point_1: f2c.Point, point_2: f2c.Point, is_gps: bool = False) -> float:
        if is_gps:
            return CoveragePlanner.f2c_point_to_location(point_1).get_distance(CoveragePlanner.f2c_point_to_location(point_2))
        return float(np.linalg.norm(CoveragePlanner.f2c_point_to_numpy(point_1) - CoveragePlanner.f2c_point_to_numpy(point_2)))

    @staticmethod
    def f2c_point_to_location(point: f2c.Point) -> Location:
        return Location(CoveragePlanner.f2c_point_to_numpy(point))

    @staticmethod
    def f2c_point_to_numpy(point: f2c.Point) -> NDArray[np.float64]:
        return np.array([point.getX(), point.getY(), point.getZ()], dtype=np.float64)
