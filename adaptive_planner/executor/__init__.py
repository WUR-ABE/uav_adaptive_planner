from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from adaptive_planner.field import Field
    from adaptive_planner.utils import CameraParameters, Waypoint


Self = TypeVar("Self", bound="Executor")  # TODO: remove python>3.11


class Executor(Protocol):
    field: Field
    output_folder: Path | None

    @classmethod
    def from_file(cls: type[Self], field: Field, output_folder: Path | None, config_file: Path) -> Self: ...

    @property
    def camera_parameters(self) -> CameraParameters: ...

    def get_image_at_coordinate(self, waypoint: Waypoint) -> NDArray[np.uint8]: ...
    def enable(self) -> None: ...
    def finish(self) -> None: ...
