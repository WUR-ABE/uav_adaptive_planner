from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from tilemapbase import Extent, Plotter, init, project
from tilemapbase.tiles import Tiles, build_OSM

from adaptive_planner import setup_logging
from adaptive_planner.executor import Executor
from adaptive_planner.utils import CameraParameters

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Literal

    from numpy.typing import NDArray

    from adaptive_planner.field import Field
    from adaptive_planner.utils import Waypoint

log = setup_logging(__name__)


class SimpleSim(Executor):
    def __init__(self, field: Field, output_folder: Path | None = None) -> None:
        self.field = field
        self.output_folder = output_folder

        init(create=True)

        self.flight_path = np.empty((0, 3), dtype=np.float64)

        self.fig, self.ax = plt.subplots(figsize=(4, 4), dpi=300)
        plt.axis("off")
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        plt.ion()  # Enable interactive mode

    @classmethod
    def from_file(cls: type[SimpleSim], field: Field, output_folder: Path | None, config_file: Path) -> SimpleSim:
        return cls(field, output_folder)

    @property
    def camera_parameters(self) -> CameraParameters:
        return CameraParameters(
            (1280, 960),
            (35.9, 24.0),
            35.0,
        )

    def get_image_at_coordinate(
        self, waypoint: Waypoint, tile_type: Literal["OSM", "ArcGis"] = "ArcGis", **kwargs: Any
    ) -> NDArray[np.uint8]:
        self.flight_path = np.vstack((self.flight_path, np.hstack((waypoint.location.gps_coordinate_lon_lat, waypoint.heading))))

        expand = 0.0005
        extent = Extent.from_lonlat(
            self.flight_path[:, 0].min() - expand,
            self.flight_path[:, 0].max() + expand,
            self.flight_path[:, 1].min() - expand,
            self.flight_path[:, 1].max() + expand,
        )

        flight_path_projected = np.array(
            [
                [*project(lon, lat), heading]
                for lon, lat, heading in zip(self.flight_path[:, 0], self.flight_path[:, 1], self.flight_path[:, 2])
            ],
            dtype=np.float64,
        )

        tiles = self.get_tiles(tile_type)

        self.ax.clear()
        plotter = Plotter(extent, tiles, height=600)
        plotter.plot(self.ax, tiles, alpha=0.8)
        self.ax.plot(flight_path_projected[:, 0], flight_path_projected[:, 1], color="blue", linewidth=expand)

        for wp in flight_path_projected:
            end_x = wp[0] + 0.1 * np.cos(wp[2] - 0.5 * np.pi)
            end_y = wp[1] + 0.1 * np.sin(wp[2] - 0.5 * np.pi)

            # Draw the axis using plt.quiver()
            plt.quiver(wp[0], wp[1], end_x - wp[0], end_y - wp[1], angles="xy", scale_units="xy", linewidth=0.1)

        self.fig.canvas.draw_idle()  # Update the plot
        plt.pause(0.001)

        return np.zeros((*np.flip(self.camera_parameters.image_size), 3), dtype=np.uint8)

    def enable(self) -> None:
        return

    def finish(self) -> None:
        plt.ioff()
        plt.show()

    @staticmethod
    def get_tiles(tile_type: Literal["OSM", "ArcGis"]) -> Tiles:
        if tile_type == "OSM":
            return build_OSM()
        elif tile_type == "ArcGis":
            ARCGIS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            return Tiles(ARCGIS_URL, "satellite")
        else:
            raise NotImplementedError(f"Tile type {tile_type} is not implemented!")
