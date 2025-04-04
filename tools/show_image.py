from __future__ import annotations

from pathlib import Path
from tap import Tap
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from adaptive_planner import setup_logging
from adaptive_planner.io import get_image_paths
from adaptive_planner.io.darwin import load_darwin_annotations
from adaptive_planner.io.topcon import read_topcon_data
from adaptive_planner.io.yolo import YoloDataset, load_yolo_annotations
from adaptive_planner.location import Location

from .metashape import get_camera_by_name, get_gsd_from_camera, gps_to_pixel_coordinate, load_project, pixel_to_gps_location

if TYPE_CHECKING:
    from typing import Any

    from Metashape import Camera, Chunk

    from adaptive_planner.utils import Annotation


log = setup_logging("show_image")


AVAILABLE_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
    (128, 0, 0),  # Maroon
]
_class_colors = {}
_class_colors_i = 0


def get_class_color(class_name: str) -> tuple[int, int, int]:
    global _class_colors, _class_colors_i

    if class_name not in _class_colors:
        _class_colors[class_name] = AVAILABLE_COLORS[_class_colors_i]
        _class_colors_i += 1

    return _class_colors[class_name]


class ImageWindow:
    def __init__(self, image_path: Path, label_path: Path) -> None:
        self.image_path = image_path
        self.label_path = label_path
        self.chunk: Chunk | None = None
        self.camera: Camera | None = None

        self.original_image = cv2.imread(str(image_path))
        self.image = self.original_image.copy()

    def show(self) -> bool:
        cv2.namedWindow(self.image_path.stem, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.image_path.stem, self.mouse_callback)
        cv2.imshow(self.image_path.stem, self.image)
        cv2.resizeWindow(self.image_path.stem, 1920, 1080)

        while True:
            key = cv2.waitKey(0)

            if key == ord(" "):
                cv2.destroyAllWindows()
                return True
            elif key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return False

    def set_metashape(self, chunk: Chunk, camera: Camera) -> None:
        self.chunk = chunk
        self.camera = camera

    def draw_annotations(self, annotations: list[Annotation]) -> None:
        for annotation in annotations:
            x_min, y_min, x_max, y_max = map(int, annotation.x1y1x2y2)
            cv2.rectangle(
                self.image,
                (x_min, y_min),
                (x_max, y_max),
                get_class_color(annotation.class_name),
                max(2, max(self.image.shape) // 1000),
            )
            cv2.putText(
                self.image,
                annotation.class_name,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                get_class_color(annotation.class_name),
                max(2, max(self.image.shape) // 2000),
            )
        log.info(f"Added {len(annotations)} bounding boxes")

    def draw_markers(self, chunk: Chunk, camera: Camera, markers: list[Location], radius: int) -> None:
        marker_coordinates = []
        marker_names = []
        marker_types = []

        for marker in markers:
            pixel_coordinates = gps_to_pixel_coordinate(chunk, camera, marker)

            if pixel_coordinates is not None:
                marker_coordinates.append(pixel_coordinates)
                marker_names.append(marker.properties["name"])
                marker_types.append(marker.properties["class_name"])

        for coordinate, marker_name, marker_type in zip(marker_coordinates, marker_names, marker_types):
            cv2.circle(
                self.image,
                coordinate,
                radius,
                get_class_color(marker_type),
                max(2, max(self.image.shape) // 1000),
            )
            cv2.putText(
                self.image,
                marker_name,
                (coordinate[0], coordinate[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                get_class_color(marker_type),
                max(2, max(self.image.shape) // 2000),
            )

        log.info(f"Added {len(marker_coordinates)} markers")

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Image coordinates: (x={x}, y={y})")
            if self.chunk is not None and self.camera is not None:
                location = pixel_to_gps_location(self.chunk, self.camera, np.array([x, y]))
                print(
                    f"GPS coordinates: (lon={location.gps_coordinate_lon_lat[0]}, lat={location.gps_coordinate_lon_lat[1]},"
                    f" alt={location.gps_coordinate_lon_lat[2]})"
                )
            print("------------")


class ArgumentError(Exception):
    pass


class ArgumentParser(Tap):
    dataset_file: Path  # Dataset file or image folder
    label_folder: Path | None = None  # YOLO label folder or Darwin annotation file
    label_file: Path | None = None  # YOLO label file
    markers_file: Path | None = None  # GPS markers file
    markers_file_crs: str = "epsg:28992"  # Input crs of the markers file
    agisoft_project_file: Path | None = None  # Path to the Agisoft Metashape project (*.psx) file
    chunk_name: str | None = None  # Name of the chunk in Agisoft
    dist_threshold: float = 0.30  # Distance threshold [m] to accept detection
    sensor_width: float = 35.9  # Sensor width of image sensor [mm] (defaults to DJI Zenmuse P1)
    subset: Literal["train", "val", "test"] = "train"  # Subset to show

    def configure(self) -> None:
        self.add_argument("dataset_file")

    def process_args(self) -> None:
        if not self.dataset_file.is_file() and not self.dataset_file.is_dir():
            raise ArgumentError(f"Dataset file {self.dataset_file.name} is not a file or folder!")

        if self.dataset_file.is_dir() and self.label_folder is None:
            raise ArgumentError("Missing required argument --label_folder")

        if self.markers_file is not None and self.agisoft_project_file is None:
            raise ArgumentError("Missing required argument --agisoft_project_file")


def main() -> None:
    args = ArgumentParser().parse_args()

    image_paths: dict[str, Path | None] = {}
    label_paths: dict[str, Path | None] = {}

    if args.dataset_file.is_file():
        yolo_dataset = YoloDataset.from_dataset_file(args.dataset_file)
        image_paths.update(yolo_dataset.image_paths)
        label_paths.update(yolo_dataset.label_paths)
        class_names = yolo_dataset.class_names
        annotations, _ = load_yolo_annotations(args.dataset_file)
    elif args.dataset_file.is_dir():
        assert args.label_folder is not None

        image_paths["train"] = args.dataset_file
        label_paths["train"] = args.label_folder

        if args.label_folder.is_file() and args.label_folder.suffix == ".zip":
            annotations, class_names = load_darwin_annotations(args.label_folder)
        elif args.label_folder.is_dir():
            class_names = None
            if args.label_file is not None:
                class_names = args.label_file.read_text().splitlines()
            annotations, class_names = load_yolo_annotations(args.dataset_file, args.label_folder, class_names=class_names)
        else:
            raise NotImplementedError("Unknown annotation format!")

        class_names = list(class_names)

    if args.markers_file is not None:
        markers = read_topcon_data(args.markers_file, crs=args.markers_file_crs)
        chunk = load_project(args.agisoft_project_file, args.chunk_name)

    image_folder = image_paths[args.subset]
    label_folder = label_paths[args.subset]
    assert image_folder is not None and label_folder is not None, f"Subset {args.subset} is not available!"

    for img_file in get_image_paths(image_folder):
        window = ImageWindow(img_file, label_folder)

        if args.markers_file is not None:
            camera = get_camera_by_name(chunk, img_file.stem)
            gsd = get_gsd_from_camera(chunk, camera, args.sensor_width)  # m/px
            radius = int(args.dist_threshold / gsd)

            window.set_metashape(chunk, camera)
            window.draw_markers(chunk, camera, markers, radius)

        image_annotation = annotations[img_file.stem]
        window.draw_annotations(image_annotation)

        if not window.show():
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
