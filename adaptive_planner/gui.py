from __future__ import annotations

from logging import ERROR, INFO, WARNING, Formatter, Handler, getLogger
from pathlib import Path
from threading import Thread
from tkinter import BOTH, DISABLED, END, NORMAL, Frame, Label, Text, Tk
from typing import TYPE_CHECKING

import numpy as np

from PIL.ImageTk import Image, PhotoImage  # type: ignore[attr-defined]
from tkintermapview import TkinterMapView

from adaptive_planner import setup_logging
from adaptive_planner.io.kml import write_gps_flightpath_to_kml, write_gps_locations_to_kml
from adaptive_planner.location import Location, switch_lon_lat
from adaptive_planner.tracking import TrackerStatus
from adaptive_planner.utils import TILE_SERVER_URL, TileServers, Waypoint

if TYPE_CHECKING:
    from logging import LogRecord

    from numpy.typing import NDArray

    from tkintermapview.canvas_path import CanvasPath
    from tkintermapview.canvas_position_marker import CanvasPositionMarker

    from adaptive_planner.executor import Executor
    from adaptive_planner.field import Field
    from adaptive_planner.planner import AdaptivePlanner


log = setup_logging(__name__)


class TkLogHandler(Handler):
    def __init__(self, text_widget: Text) -> None:
        super().__init__()

        self.text_widget = text_widget
        self.create_tag_styles()

    def create_tag_styles(self) -> None:
        self.text_widget.tag_configure("info", foreground="green")
        self.text_widget.tag_configure("warning", foreground="orange")
        self.text_widget.tag_configure("error", foreground="red")

    def emit(self, record: LogRecord) -> None:
        msg = self.format(record) + "\n"
        self.text_widget.config(state=NORMAL)

        if record.levelno == INFO:
            self.text_widget.insert(END, msg, "info")
        elif record.levelno == WARNING:
            self.text_widget.insert(END, msg, "warning")
        elif record.levelno == ERROR:
            self.text_widget.insert(END, msg, "error")
        else:
            self.text_widget.insert(END, msg)

        self.text_widget.see(END)
        self.text_widget.config(state=DISABLED)


class WorkerThread(Thread):
    def __init__(self, adaptive_planner: AdaptivePlanner, only_inspection: bool = False) -> None:
        super().__init__()

        self.adaptive_planner = adaptive_planner
        self.only_inspection = only_inspection

        if only_inspection and self.adaptive_planner.drone_waypoint_callback is not None:
            self.adaptive_planner.drone_waypoint_callback(
                Waypoint(
                    self.adaptive_planner.executed_flight_path[-1],
                    self.adaptive_planner.executed_flight_path[-2].get_heading(self.adaptive_planner.executed_flight_path[-1]),
                ),
                self.adaptive_planner.executed_flight_path,
            )

        if only_inspection and self.adaptive_planner.detection_callback is not None:
            for loc in self.adaptive_planner.tracker.get_locations(status_mask=TrackerStatus.ACCEPTED | TrackerStatus.INVESTIGATED):
                self.adaptive_planner.detection_callback(loc)

    def run(self) -> None:
        if self.only_inspection:
            self.adaptive_planner.execute_inspection(
                self.adaptive_planner.tracker.get_locations(status_mask=TrackerStatus.TO_BE_INVESTIGATED),
                self.adaptive_planner.executed_flight_path[-1],
                None,
            )
        else:
            self.adaptive_planner.execute()

        if self.adaptive_planner.output_folder:
            write_gps_flightpath_to_kml(
                self.adaptive_planner.output_folder / "flight_path.kml",
                self.adaptive_planner.executed_flight_path,
                f"{self.adaptive_planner.output_folder} flight path",
            )
            log.info(f"Saved flight path to {self.adaptive_planner.output_folder / 'flight_path.kml'}")

            write_gps_locations_to_kml(
                self.adaptive_planner.output_folder / "detections.kml",
                self.adaptive_planner.detected_objects,
            )
            log.info(f"Saved detected objects to {self.adaptive_planner.output_folder / 'detections.kml'}")


class AdaptivePlannerGui(Tk):
    def __init__(
        self,
        executor: Executor,
        field: Field,
        adaptive_planner: AdaptivePlanner,
        initial_zoom: int = 20,
        tile_server: TileServers = TileServers.ARCGIS,
    ):
        super().__init__()
        self.title("Adaptive planner")
        self.geometry("1235x960")

        self.update_idletasks()

        self._tile_server = tile_server
        self._initial_zoom = initial_zoom

        self.executor = executor
        self.field = field
        self.adaptive_planner = adaptive_planner

        self.setup_ui()

        self.start_location = None if field.start_location is None else field.start_location.gps_coordinate_lat_lon[:2]
        if self.start_location is None:
            self.start_location = switch_lon_lat(np.array(field.boundary.exterior.coords[0], dtype=np.float64))

        self.map_widget.set_position(*self.start_location)

        # Plot field boundaries
        polygon = [[c[1], c[0]] for c in field.boundary.exterior.coords]
        self.map_widget.set_polygon(polygon, outline_color="yellow", fill_color=None)

        self._map_widget_waypoint_marker: CanvasPositionMarker | None = None
        self._map_widget_flight_path: CanvasPath | None = None
        self._map_widget_base_path: CanvasPath | None = None

        self.update_idletasks()

    def setup_ui(self) -> None:
        self.map_widget = TkinterMapView(self, width=470, height=470)
        self.map_widget.place(x=20, y=20)
        self.map_widget.set_tile_server(TILE_SERVER_URL[self._tile_server])
        self.map_widget.set_zoom(self._initial_zoom)

        self.image_frame = Frame(self, width=705, height=470)
        self.image_frame.pack_propagate(False)
        self.image_frame.place(x=500, y=20)

        self.image_view = Label(self.image_frame, bg="#d3d3d3")
        self.image_view.pack(fill=BOTH, expand=True)

        log_frame = Frame(self, height=430, width=1195)
        log_frame.pack_propagate(False)
        log_frame.place(x=20, y=510)

        log_text = Text(log_frame)
        log_text.pack(fill=BOTH, expand=True)

        self.logging_handler = TkLogHandler(log_text)
        formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.logging_handler.setFormatter(formatter)
        getLogger().addHandler(self.logging_handler)
        getLogger().setLevel(INFO)

    def start_task(self, only_inspection: bool = False) -> None:
        self.worker_thread = WorkerThread(self.adaptive_planner, only_inspection=only_inspection)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def update_drone_waypoint(self, waypoint: Waypoint, flight_path: list[Location]) -> None:
        if self._map_widget_waypoint_marker is None:
            icon_path = Path(__file__).parent / "plane.png"
            icon = PhotoImage(Image.open(icon_path).rotate(-np.rad2deg(waypoint.heading)).resize((25, 25)))  # type: ignore [attr-defined]
            self._map_widget_waypoint_marker = self.map_widget.set_marker(*waypoint.location.gps_coordinate_lat_lon[:2], icon=icon)

        if self._map_widget_flight_path is None and len(flight_path) > 0:
            self._map_widget_flight_path = self.map_widget.set_path(
                [self.start_location] + [wp.gps_coordinate_lat_lon[:2] for wp in flight_path],
                color="#8b0000",
                width=3,
            )

        icon_path = Path(__file__).parent / "plane.png"
        icon = PhotoImage(Image.open(icon_path).rotate(-np.rad2deg(waypoint.heading)).resize((25, 25)))  # type: ignore [attr-defined]
        self._map_widget_waypoint_marker.set_position(*waypoint.location.gps_coordinate_lat_lon[:2])
        self._map_widget_waypoint_marker.change_icon(icon)

        if self._map_widget_flight_path is not None and len(flight_path) > 0:
            self._map_widget_flight_path.set_position_list([self.start_location] + [wp.gps_coordinate_lat_lon[:2] for wp in flight_path])

        # Reset map to drone position
        self.map_widget.set_position(*waypoint.location.gps_coordinate_lat_lon[:2])

    def update_base_flight_path(self, base_flight_path: list[Waypoint]) -> None:
        if self._map_widget_base_path is None:
            self._map_widget_base_path = self.map_widget.set_path(
                [wp.location.gps_coordinate_lat_lon[:2] for wp in base_flight_path],
                width=2,
            )
            return

        self._map_widget_base_path.set_position_list([wp.location.gps_coordinate_lat_lon[:2] for wp in base_flight_path])

    def update_detection_image(self, detection_image: NDArray[np.uint8]) -> None:
        image = Image.fromarray(detection_image[:, :, ::-1])  # type: ignore [attr-defined]
        image = self.resize_to_fit(image, self.image_frame.winfo_width(), self.image_frame.winfo_height())
        tk_image = PhotoImage(image)

        self.image_view.config(image=tk_image)

        # Keep a reference to prevent garbage collection
        self.image_view.image = tk_image  # type: ignore [attr-defined]

    def add_detection(self, detection: Location) -> None:
        for mrk in self.map_widget.canvas_marker_list:
            if np.array_equal(mrk.position, detection.gps_coordinate_lat_lon[:2]):
                break
        else:
            icon_path = Path(__file__).parent / "marker.png"
            icon = PhotoImage(Image.open(icon_path).resize((10, 10)))  # type: ignore [attr-defined]
            self.map_widget.set_marker(*detection.gps_coordinate_lat_lon[:2], icon=icon)

    @staticmethod
    def resize_to_fit(image: Image, target_width: int, target_height: int) -> Image:
        width, height = image.size
        aspect_ratio = width / height

        if width > height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_width = int(target_height * aspect_ratio)
            new_height = target_height

        resized_image = image.resize((new_width, new_height))
        return resized_image
