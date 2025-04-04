from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any

    from adaptive_planner.field import Field
    from adaptive_planner.location import Location
    from adaptive_planner.utils import CameraParameters, Waypoint


class Planner(Protocol):
    field: Field
    camera_parameters: CameraParameters

    @property
    def config_dict(self) -> dict[str, Any]: ...

    def plan(self, start_position: Location, altitude: float, output_file: Path | None = None) -> list[Waypoint]: ...
    def set_objects_and_end_location(self, locations: list[Location], end_location: Location | None) -> None: ...
