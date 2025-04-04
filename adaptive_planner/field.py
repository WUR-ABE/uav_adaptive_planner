from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING
from yaml import safe_load as load_yaml

import numpy as np

from adaptive_planner.io.geojson import read_polygon_file
from adaptive_planner.location import Location

if TYPE_CHECKING:
    from typing import Any

    from shapely import Polygon


@dataclass
class Field:
    name: str
    boundary_file: Path
    start_location: Location | None
    orthomosiac_scheme_file: Path | None
    locations_to_ignore: list[Location] = field(default_factory=list)

    @cached_property
    def boundary(self) -> Polygon:
        return read_polygon_file(self.boundary_file)

    @classmethod
    def from_file(cls: type[Field], file: Path) -> Field:
        with file.open("r") as file_stream:
            data: dict[str, Any] = load_yaml(file_stream)

            locations_to_ignore = []
            for i, coordinates in enumerate(data.get("locations_to_ignore", [])):
                locations_to_ignore.append(Location(np.array(coordinates, dtype=np.float64), properties={"name": f"ignore_location_{i}"}))

            start_location = (
                None
                if data["start_position"] is None
                else Location(
                    np.array(data["start_position"], dtype=np.float64),
                    properties={"name": f"{data['name']}_start_location"},
                )
            )

            return cls(
                data["name"],
                Path(data["boundary_file"]),
                start_location,
                None if data["orthomosaic_scheme_file"] is None else Path(data["orthomosaic_scheme_file"]),
                locations_to_ignore=locations_to_ignore,
            )
