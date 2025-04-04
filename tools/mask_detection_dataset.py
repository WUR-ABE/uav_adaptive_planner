from __future__ import annotations

from pathlib import Path
from tap import Tap
from typing import TYPE_CHECKING

import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from adaptive_planner import setup_logging
from adaptive_planner.io.topcon import read_topcon_data
from adaptive_planner.io.yolo import YoloDataset
from adaptive_planner.location import Location

from .metashape import (
    get_camera_by_name,
    get_gsd_from_camera,
    get_image_size_from_camera,
    get_visible_locations,
    gps_to_pixel_coordinate,
    load_project,
    pixel_to_gps_location,
)

if TYPE_CHECKING:
    from Metashape import Camera, Chunk

    from adaptive_planner.utils import Annotation

log = setup_logging("create_train_val_dataset")


def mask_labels_from_image(
    input_path: Path,
    output_path: Path,
    labels: list[Annotation],
    mask_color: tuple[int, int, int] = (158, 179, 181),
    mask_inflation: float = 1.1,
) -> None:
    img = cv2.imread(str(input_path))

    for annotation in labels:
        new_w = int(annotation.w * mask_inflation)
        new_h = int(annotation.h * mask_inflation)
        delta_w = (new_w - int(annotation.w)) // 2
        delta_h = (new_h - int(annotation.h)) // 2

        new_x = int(annotation.x) - delta_w
        new_y = int(annotation.y) - delta_h

        cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), mask_color, thickness=-1)
    cv2.imwrite(str(output_path), img)


def mask_all_annotations_without_marker(
    input_dataset: YoloDataset,
    output_datset: YoloDataset,
    images: list[Path],
    markers: list[Location],
    project: Chunk,
    max_dist: float = 0.30,
    mask_color: tuple[int, int, int] = (158, 179, 181),
    change_class_name: bool = False,
    ignore_errors: bool = False,
    dry_run: bool = False,
) -> None:
    for image_path in tqdm(images, desc="mask images"):
        camera_name = image_path.stem
        if camera_name.startswith("train_"):
            camera_name = camera_name[len("train_") :]
        elif camera_name.startswith("val_"):
            camera_name = camera_name[len("val_") :]

        camera = get_camera_by_name(project, camera_name)
        labels = input_dataset.get_label(image_path.stem)
        visible_markers = get_visible_locations(project, camera, markers)
        label_locations = [pixel_to_gps_location(project, camera, lbl.coordinate) for lbl in labels]

        distance_matrix = np.zeros((len(labels), len(visible_markers)), dtype=np.float32)

        for i, label_location in enumerate(label_locations):
            for j, marker_location in enumerate(visible_markers):
                distance = label_location.get_distance(marker_location)
                distance_matrix[i, j] = distance if distance <= max_dist else 1e7  # just really high value

        lbl_i, marker_j = linear_sum_assignment(distance_matrix)

        labels_to_keep = []
        labels_to_remove = []
        for i, j in zip(lbl_i, marker_j):
            if labels[i].class_name != visible_markers[j].properties["class_name"]:
                log.warning(
                    f"Marker {visible_markers[j].properties['name']} (x={labels[i].x}, y={labels[i].y}) has annotation class"
                    f" '{labels[i].class_name}' while marker has class '{visible_markers[j].properties['class_name']}' in"
                    f" {image_path.stem}!"
                )
                if change_class_name:
                    log.warning("Changing class name")
                    labels[i].class_name = visible_markers[j].properties["class_name"]

            # TP should stay in the dataset
            if float(distance_matrix[i, j]) <= max_dist:
                labels_to_keep.append(labels[i])

            # FP and FN
            else:
                if marker_close_to_border(project, camera, visible_markers[j], max_dist):
                    labels_to_remove.append(labels[i])
                    continue

                distance = label_locations[i].get_distance(visible_markers[j])
                log.fatal(
                    f"Distance between annotation at {labels[i].yolo_xywh[:2]} and gps location {visible_markers[j].properties['name']} is"
                    f" too large ({distance}m) in {image_path.stem}!"
                )
                if not dry_run:
                    raise RuntimeError("Distance too large!")

        # FP should be masked out
        for i in range(len(label_locations)):
            if i not in lbl_i:
                labels_to_remove.append(labels[i])

        # FN should not happen except when markers are close to the image border
        for j in range(len(visible_markers)):
            if j not in marker_j:
                if marker_close_to_border(project, camera, visible_markers[j], max_dist):
                    continue

                log.fatal(
                    f"Could find backprojection of {visible_markers[j].properties['name']} close to the original position in"
                    f" {image_path.stem}!"
                )

                if not dry_run and not ignore_errors:
                    raise RuntimeError("Backprojection close to original position")

        assert len(labels_to_keep) + len(labels_to_remove) == len(labels)

        if not dry_run:
            if image_path in input_dataset.get_train_images():
                output_img_path = output_datset.train_image_folder / image_path.name
            elif image_path in input_dataset.get_validation_images():
                output_img_path = output_datset.validation_image_folder / image_path.name
            else:
                raise RuntimeError(f"Could not find dataset for image {image_path.stem}")

            mask_labels_from_image(image_path, output_img_path, [l for l in labels if l not in labels_to_keep], mask_color=mask_color)
            output_datset.write_label_file(image_path.stem, labels_to_keep)


def marker_close_to_border(project: Chunk, camera: Camera, marker: Location, max_dist: float) -> bool:
    x, y = gps_to_pixel_coordinate(project, camera, marker)
    width, height = get_image_size_from_camera(camera)
    gsd = get_gsd_from_camera(project, camera, 35.9, altitude=marker.gps_coordinate_lat_lon[2])
    distance_border = np.array([x, width - x - 1, y, height - y - 1], dtype=np.float32) * gsd

    return distance_border.min() <= max_dist  # type: ignore[no-any-return]


class ArgumentError(Exception):
    pass


class ArgumentParser(Tap):
    input_dataset: Path  # Path to the YOLO dataset file or folder
    output_dataset: Path  # Folder to the outputd dataset
    train_markers_file: Path  # Path to the train markers file
    val_markers_file: Path  # Path to the validation markers file
    agisoft_project_file: Path  # Path to Agisoft project file
    markers_file_crs: str = "epsg:28992"  # Input crs of the markers files
    chunk_name: str | None = None  # Name of the chunk in Agisoft
    max_dist: float = 0.3  # Maximum distance between marker and backprojected marker in m
    mask_color: tuple[int, int, int] = (158, 179, 181)  # BGR color of the applied mask
    change_class_name: bool = False  # Change class name of annotation to the class name of the marker when different
    ignore_errors: bool = False  # Ignore errors
    dry_run: bool = False  # Check if dataset is valid without writing data

    def configure(self) -> None:
        self.add_argument("input_dataset")
        self.add_argument("output_dataset")
        self.add_argument("train_markers_file")
        self.add_argument("val_markers_file")
        self.add_argument("agisoft_project_file")
        self.add_argument("--mask_color", nargs=3)

    def process_args(self) -> None:
        if not self.input_dataset.is_file() and not self.input_dataset.is_dir():
            raise ArgumentError(f"Input dataset {self.input_dataset.name} is not a file and not a directory!")

        if not self.train_markers_file.is_file():
            raise FileNotFoundError(f"Train markers file {self.train_markers_file.name} does not exist!")

        if not self.val_markers_file.is_file():
            raise FileNotFoundError(f"Validation markers file {self.val_markers_file.name} does not exist!")

        if not self.agisoft_project_file.is_file():
            raise FileNotFoundError(f"Agisoft project file {self.agisoft_project_file.name} does not exist!")


def main() -> None:
    args = ArgumentParser().parse_args()

    if args.input_dataset.is_file():
        input_dataset = YoloDataset.from_dataset_file(args.input_dataset)
    elif args.input_dataset.is_dir():
        input_dataset = YoloDataset.from_dataset_file(args.input_dataset / "data.yml")

    if args.output_dataset.is_file():
        output_dataset = YoloDataset.from_dataset_file(args.output_dataset)
    elif args.output_dataset.is_dir():
        output_dataset = YoloDataset.from_dataset_file(args.output_dataset / "data.yml")
    else:
        output_dataset = YoloDataset.from_root(args.output_dataset, input_dataset.class_names)
        output_dataset.initialise_dataset()
        output_dataset.write_dataset_file()
        output_dataset.write_classes_file()

    train_markers = read_topcon_data(args.train_markers_file, args.markers_file_crs)
    val_markers = read_topcon_data(args.val_markers_file, args.markers_file_crs)

    project = load_project(args.agisoft_project_file, chunk_name=args.chunk_name)

    if project is None:
        raise RuntimeError("Could not generate train/val dataset!")

    mask_all_annotations_without_marker(
        input_dataset,
        output_dataset,
        input_dataset.get_train_images(),
        train_markers,
        project,
        max_dist=args.max_dist,
        mask_color=args.mask_color,
        change_class_name=args.change_class_name,
        dry_run=args.dry_run,
        ignore_errors=args.ignore_errors,
    )

    mask_all_annotations_without_marker(
        input_dataset,
        output_dataset,
        input_dataset.get_validation_images(),
        val_markers,
        project,
        max_dist=args.max_dist,
        mask_color=args.mask_color,
        change_class_name=args.change_class_name,
        dry_run=args.dry_run,
        ignore_errors=args.ignore_errors,
    )


if __name__ == "__main__":
    main()
