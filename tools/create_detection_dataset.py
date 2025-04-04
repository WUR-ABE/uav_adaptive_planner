from __future__ import annotations

from pathlib import Path
from shutil import copy2
from tap import Tap
from typing import TYPE_CHECKING

import numpy as np

from tqdm.auto import tqdm

from adaptive_planner import setup_logging
from adaptive_planner.io.topcon import read_topcon_data, write_topcon_data
from adaptive_planner.io.yolo import YoloDataset

from .metashape import get_annotations_by_camera, get_cameras_by_location, load_project

if TYPE_CHECKING:
    from Metashape import Camera, Chunk

    from adaptive_planner.location import Location

log = setup_logging("create_train_val_dataset")


def get_train_val_markers(markers: list[Location], p_train: float, p_val: float) -> tuple[list[Location], list[Location]]:
    class_labels = np.array([m.properties["class_name"] for m in markers])
    classes, counts = np.unique(class_labels, return_counts=True)
    train_markers: list[Location] = []
    val_markers: list[Location] = []

    for i in range(len(classes)):
        indices = np.where(class_labels == classes[i])[0]
        np.random.shuffle(indices)

        n_train = int(round(counts[i] * p_train))
        n_val = int(round(counts[i] * p_val))

        train_markers.extend(markers[j] for j in indices[:n_train])
        val_markers.extend(markers[j] for j in indices[n_train : n_train + n_val])

    return train_markers, val_markers


def create_dataset(
    dataset: YoloDataset,
    image_folder: Path,
    project: Chunk,
    cameras: dict[str, Camera],
    markers: list[Location],
    annotation_size: tuple[float, float] = (0.1, 0.1),
    prefix: str = "",
) -> None:
    for name, camera in tqdm(cameras.items(), "creating dataset"):
        image_path = Path(camera.photo.path)
        name = prefix + name

        if not image_path.is_file():
            raise FileNotFoundError(f"Cannot find image {camera.photo.path}!")

        # Only copy when not existing
        if not (image_folder / image_path.name).is_file():
            copy2(image_path, image_folder / (name + image_path.suffix))
            dataset.write_label_file(name, get_annotations_by_camera(project, camera, markers, annotation_size))


class ArgumentError(Exception):
    pass


class ArgumentParser(Tap):
    markers_file: Path  # GPS markers file
    output_folder: Path  # Output folder for the YOLO dataset
    agisoft_project_file: Path  # Path to Agisoft project file
    train_val_split: tuple[float, float] = (0.1, 0.05)  # Percentage train and validation markers (0-1)
    markers_file_crs: str = "epsg:28992"  # Input crs of the markers file
    chunk_name: str | None = None  # Name of the chunk in Agisoft
    annotation_size: tuple[float, float] = (0.1, 0.1)  # Size of bounding box in meters

    def configure(self) -> None:
        self.add_argument("markers_file")
        self.add_argument("output_folder")
        self.add_argument("agisoft_project_file")
        self.add_argument("--train_val_split", nargs=2)
        self.add_argument("--annotation_size", nargs=2)

    def process_args(self) -> None:
        if not self.agisoft_project_file.is_file():
            raise FileNotFoundError(f"Agisoft project file {self.agisoft_project_file.name} does not exist!")

        if not self.markers_file.is_file():
            raise FileNotFoundError(f"Markers file {self.markers_file.name} does not exist!")

        if sum(self.train_val_split) > 1.0:
            raise ArgumentError("Total of train and val split cannot be larger than 1!")

        if self.output_folder.is_dir() and not (self.output_folder / "data.yaml").is_file():
            raise ArgumentError(
                f"Output folder '{self.output_folder.name}'already exist but is not a YOLO dataset. Select an empty folder or YOLO dataset."
            )


def main() -> None:
    args = ArgumentParser().parse_args()

    project = load_project(args.agisoft_project_file, chunk_name=args.chunk_name)

    if project is None:
        raise RuntimeError("Could not generate train/val dataset!")


    all_markers = read_topcon_data(args.markers_file, crs=args.markers_file_crs)

    if (args.markers_file.parent / "train_markers.csv").is_file() and (args.markers_file.parent / "val_markers.csv").is_file():
        log.info(
            "\u2757Train and validation markers are already defined, using them instead. Delete 'train_markers.csv' and 'val_markers.csv'"
            " to create new train/val split."
        )
        train_markers = read_topcon_data(args.markers_file.parent / "train_markers.csv", crs=args.markers_file_crs)
        val_markers = read_topcon_data(args.markers_file.parent / "val_markers.csv", crs=args.markers_file_crs)
    else:
        train_markers, val_markers = get_train_val_markers(all_markers, *args.train_val_split)

        # Write train and validation to markers file
        write_topcon_data(train_markers, args.markers_file.parent / "train_markers.csv", crs=args.markers_file_crs)
        write_topcon_data(val_markers, args.markers_file.parent / "val_markers.csv", crs=args.markers_file_crs)

    if args.output_folder.is_dir():
        dataset = YoloDataset.from_dataset_file(args.output_folder / "data.yml")
    else:
        # Create new dataset
        class_names = list({m.properties["class_name"] for m in all_markers})
        dataset = YoloDataset.from_root(args.output_folder, class_names)
        dataset.initialise_dataset()

    train_cameras = {camera.label: camera for marker in train_markers for camera in get_cameras_by_location(project, marker)}
    val_cameras = {camera.label: camera for marker in val_markers for camera in get_cameras_by_location(project, marker)}

    if len(train_cameras) == 0 or len(val_cameras) == 0:
        raise RuntimeError("Could not find markers any image! Check markers file.")

    create_dataset(
        dataset, dataset.train_image_folder, project, train_cameras, all_markers, annotation_size=args.annotation_size, prefix="train_"
    )
    create_dataset(
        dataset, dataset.validation_image_folder, project, val_cameras, all_markers, annotation_size=args.annotation_size, prefix="val_"
    )

    log.info("Done")


if __name__ == "__main__":
    main()
