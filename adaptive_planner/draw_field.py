from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from tempfile import NamedTemporaryFile
from tkinter import Tk
from typing import TYPE_CHECKING
from yaml import safe_dump

import numpy as np

from shapely import Polygon, coverage_union_all
from tkintermapview import TkinterMapView
from tkintermapview.canvas_button import CanvasButton

from adaptive_planner import setup_logging
from adaptive_planner.field import Field
from adaptive_planner.io.geojson import write_polygon_file
from adaptive_planner.io.orthomosaic import get_tile_polygons
from adaptive_planner.location import Location
from adaptive_planner.utils import TILE_SERVER_URL, TileServers

if TYPE_CHECKING:
    from tkintermapview.canvas_position_marker import CanvasPositionMarker


class Stage(Enum):
    SELECT_AOI = auto()
    SELECT_START_POINT = auto()


log = setup_logging(__name__)


class DrawFieldGUI(Tk):
    """
    GUI to select the field border and initial position of the drone. Internally, this class uses lat-lon notation since
    tkintermapview uses this format.
    """

    WIDTH = 1000
    HEIGHT = 700

    def __init__(
        self,
        initial_position_lon_lat: tuple[float, float],
        initial_zoom: int = 20,
        tile_server: TileServers = TileServers.ARCGIS,
        select_start_location: bool = True,
        scheme_file: Path | None = None,
    ) -> None:
        super().__init__()

        self.title("Select flight area")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")

        self._tile_server = tile_server
        self._initial_zoom = initial_zoom

        self.map_widget = TkinterMapView(self, width=1000, height=700, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)

        self.map_widget.set_position(initial_position_lon_lat[1], initial_position_lon_lat[0])
        self.map_widget.set_zoom(self._initial_zoom)
        self.map_widget.set_tile_server(TILE_SERVER_URL[self._tile_server])
        self.map_widget.add_left_click_map_command(self.left_click_event)

        self.scheme_file_polygon = None
        if scheme_file is not None:
            polygon = coverage_union_all([self.polygon_to_gps(p) for p in get_tile_polygons(scheme_file).values()])
            self.scheme_file_polygon = np.array([[*c] for c in polygon.exterior.coords], dtype=np.float64)  # type: ignore[attr-defined]
            self.map_widget.set_polygon(self.scheme_file_polygon, outline_color="yellow", fill_color=None)

        self.continue_button = CanvasButton(self.map_widget, (self.WIDTH - 80, 20), text="→", command=self.next)

        self._select_start_position = select_start_location
        self._polygon_coordinates: list[tuple[float, float]] = []
        self._start_point: tuple[float, float] | None = None
        self._current_stage = Stage.SELECT_AOI
        self._do_not_process_click = False

    def next(self) -> None:
        self._do_not_process_click = True

        if self._current_stage == Stage.SELECT_AOI:
            if len(self._polygon_coordinates) >= 3:
                if not self._select_start_position:
                    self.quit()
                    return

                self._current_stage = Stage.SELECT_START_POINT
                self.title("Select start point")
                return

            log.error("Need to select at least 3 points!")

        elif self._current_stage == Stage.SELECT_START_POINT:
            if self._start_point is not None:
                self.quit()
                return

            log.error("Need to select a start point!")

    def draw_polygon(self) -> None:
        self.map_widget.delete_all_polygon()
        self.map_widget.delete_all_marker()

        if self.scheme_file_polygon is not None:
            self.map_widget.set_polygon(self.scheme_file_polygon, outline_color="yellow", fill_color=None)

        for coordinate in self._polygon_coordinates:
            self.map_widget.set_marker(
                *coordinate,
                command=self.marker_click_event,
                marker_color_circle="#9B261E",
                marker_color_outside="#C5542D",
            )

        if self._start_point is not None:
            self.map_widget.set_marker(
                *self._start_point,
                command=self.marker_click_event,
                marker_color_circle="#627ba4",
                marker_color_outside="#0099ff",
            )

        if len(self._polygon_coordinates) >= 3:
            self.map_widget.set_polygon(
                self._polygon_coordinates,
                fill_color=None,
                outline_color="blue",
                border_width=2,
                name="flight_area",
            )

    def left_click_event(self, coordinates: tuple[float, float]) -> None:
        if self._current_stage == Stage.SELECT_AOI:
            if not self._do_not_process_click:
                self._polygon_coordinates.append(coordinates)
                self.draw_polygon()

        elif self._current_stage == Stage.SELECT_START_POINT:
            if not self._do_not_process_click:
                self._start_point = coordinates
                self.draw_polygon()

        self._do_not_process_click = False

    def marker_click_event(self, marker: CanvasPositionMarker) -> None:
        if self._current_stage == Stage.SELECT_AOI:
            self._polygon_coordinates.remove(marker.position)
            self.draw_polygon()

        self._do_not_process_click = True

    def to_field(self, folder: Path | None = None, name: str = "field", scheme_file: Path | None = None) -> Field | None:
        if len(self._polygon_coordinates) < 3 or self._start_point is None:
            return None

        if folder is None:
            boundary_file_path = Path(NamedTemporaryFile(suffix=".geojson").name)
            config_file_path = Path(NamedTemporaryFile(suffix=".yaml").name)
        else:
            boundary_file_path = folder / (name + ".geojson")
            config_file_path = folder / (name + ".yaml")

        polygon = Polygon(np.array(self._polygon_coordinates, dtype=np.float64).squeeze())
        write_polygon_file(boundary_file_path, polygon, name)

        config_data = {
            "name": name,
            "boundary_file": str(boundary_file_path),
            "start_position": np.flip(np.array(self._start_point, dtype=np.float64)).tolist(),
            "orthomosaic_scheme_file": None if scheme_file is None else str(scheme_file),
        }

        with config_file_path.open("w") as config_file_stream:
            safe_dump(config_data, config_file_stream, indent=2)

        return Field(
            name,
            boundary_file_path,
            Location(np.flip(np.array(self._start_point, dtype=np.float64))),
            scheme_file,
        )

    @staticmethod
    def polygon_to_gps(polygon: Polygon) -> Polygon:
        coords_gps = []
        for coord in polygon.exterior.coords:
            location = Location.from_utm(np.array(coord, dtype=np.float64))
            coords_gps.append(location.gps_coordinate_lat_lon[:2])
        return Polygon(coords_gps)
