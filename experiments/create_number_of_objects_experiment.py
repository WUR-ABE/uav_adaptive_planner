from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from tap import Tap
from typing import TYPE_CHECKING
from yaml import dump

import cv2
import numpy as np

from pyproj import CRS
from shapely import Point
from tqdm.auto import tqdm

from adaptive_planner import setup_logging
from adaptive_planner.field import Field
from adaptive_planner.io.kml import write_gps_locations_to_kml
from adaptive_planner.io.orthomosaic import get_tile_polygons, load_tiles
from adaptive_planner.io.topcon import read_topcon_data
from adaptive_planner.io.yolo import YoloDataset
from adaptive_planner.location import Location
from adaptive_planner.utils import LogProcessingTime, polygon_to_crs

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

    from shapely import Polygon


log = setup_logging(__name__)


@dataclass
class Cluster:
    mean: Point
    cov: NDArray[np.float32]
    n: int

    mean_np: NDArray[np.float64] = field(init=False)
    n_drawn: int = field(init=False)

    def __post_init__(self) -> None:
        self.mean_np = np.array([self.mean.x, self.mean.y], dtype=np.float64)
        self.n_drawn = 0

    def get_point(
        self,
        field_polygon: Polygon,
        existing_points: list[Point],
        min_distance: float = 1.0,
        rng: Generator = np.random.default_rng(),
    ) -> Point | None:
        if self.n_drawn > self.n:
            return None

        while True:
            p = Point(rng.multivariate_normal(self.mean_np, self.cov, size=1))

            if field_polygon.contains(p) and all(p.distance(loc) >= min_distance for loc in existing_points):
                self.n_drawn += 1
                return p


class DistributionType(Enum):
    UNIFORM = auto()
    CLUSTERED = auto()


def get_uniform_positions(
    field_polygon: Polygon,
    existing_points: list[Point],
    n: int,
    min_distance: float = 1.0,
    rng: Generator = np.random.default_rng(),
) -> list[Point]:
    random_points: list[Point] = []

    minx, miny, maxx, maxy = field_polygon.bounds
    while len(random_points) < n:
        pt = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if field_polygon.contains(pt) and all(pt.distance(loc) >= min_distance for loc in existing_points + random_points):
            random_points.append(pt)

    return random_points


def get_clustered_positions(
    field_polygon: Polygon,
    existing_points: list[Point],
    n: int,
    min_distance: float = 1.0,
    rng: Generator = np.random.default_rng(),
) -> list[Point]:
    clustered_points: list[Point] = []

    cluster: Cluster | None = None
    while len(clustered_points) < n:
        if cluster is None:
            mean = get_uniform_positions(field_polygon, existing_points, 1, min_distance=15.0, rng=rng)[0]
            cov = rng.normal(5.0, 2.0, size=(2, 2))
            cov = np.dot(cov, cov.T)  # Make positive semidefinite
            n_cluster = round(rng.normal(8, 3))

            cluster = Cluster(mean, cov, n_cluster)  # type: ignore[arg-type]

        if (point := cluster.get_point(field_polygon, existing_points + clustered_points, min_distance=min_distance, rng=rng)) is None:
            cluster = None
            continue

        clustered_points.append(point)

    return clustered_points


def get_objects_in_field(
    field: Field,
    dataset: YoloDataset,
    existing_locations: list[Location],
    n: int,
    distribution: DistributionType,
    min_distance: float = 1.0,
    rng: Generator = np.random.default_rng(),
) -> list[Location]:
    assert field.orthomosiac_scheme_file is not None, "Orthomosaic scheme file should be defined!"

    rng.shuffle(existing_locations)  # type: ignore[arg-type]
    if n <= len(existing_locations):
        return existing_locations[:n]

    # Load first tile to get the CRS
    tile_polygons = get_tile_polygons(field.orthomosiac_scheme_file)
    crs_epsg = load_tiles([list(tile_polygons.keys())[0]]).crs.to_epsg()

    field_polygon = polygon_to_crs(field.boundary, CRS.from_epsg(crs_epsg))

    with LogProcessingTime(log, "Calculating random points"):
        if distribution == DistributionType.UNIFORM:
            points = get_uniform_positions(
                field_polygon,
                [p.to_point() for p in existing_locations],
                n - len(existing_locations),
                min_distance=min_distance,
                rng=rng,
            )
        elif distribution == DistributionType.CLUSTERED:
            points = get_clustered_positions(
                field_polygon,
                [p.to_point() for p in existing_locations],
                n - len(existing_locations),
                min_distance=min_distance,
                rng=rng,
            )
        else:
            raise NotImplementedError(f"Distribution {distribution} is not implemented!")

    with LogProcessingTime(log, "Creating objects"):
        for p in tqdm(points, desc="Generating new objects"):
            location = Location.from_point(p)

            image, class_name = get_random_object_image(dataset, rng=rng)
            angle = rng.uniform(0.0, 2 * np.pi, 1).item()

            location.properties["name"] = f"additional_object_{len(existing_locations)}"
            location.properties["class_name"] = class_name
            location.properties["object_image"] = image
            location.properties["object_image_angle"] = angle
            existing_locations.append(location)

    return existing_locations


def get_random_object_image(dataset: YoloDataset, rng: Generator = np.random.default_rng()) -> tuple[NDArray[np.uint8], str]:
    while True:
        image_path = dataset.get_train_images()[rng.choice(dataset.get_num_train_images(), 1).item()]
        labels = dataset.get_label(image_path.stem)

        if len(labels) > 0:
            label = labels[rng.choice(len(labels), 1).item()]
            image = cv2.imread(str(image_path))

            # Explicitly copy image to avoid holding the complete large image in memory
            cropped_image = image[int(label.y) : int(label.y + label.h), int(label.x) : int(label.x + label.w)].copy()
            cropped_image = cropped_image.astype(np.uint8)[..., ::-1]  # BGR -> RGB
            return cropped_image, label.class_name


def write_executor_config(output_file: Path, gt_file: Path, locations_file: Path, n: int, seed: int | None = None) -> None:
    data = {
        "override_number_of_objects": n,
        "gt_file": str(gt_file),
        "override_locations_file": str(locations_file),
    }

    if seed is not None:
        data["seed"] = seed

    with output_file.open("w") as f:
        dump(data, f, indent=2)


class ArgumentParser(Tap):
    experiment_folder: Path  # Path to the experiment folder
    dataset_file: Path  # Path to the detection dataset file
    field: Path  # Path to the field file
    gt_file: Path  # Path to ground truth locations
    distribution: DistributionType  # Type of distribution
    range: list[int]  # Range of number of objects (min, max, interval)
    seed: int | None = None  # Random seed

    def configure(self) -> None:
        self.add_argument("experiment_folder")
        self.add_argument("dataset_file")
        self.add_argument("field")
        self.add_argument("gt_file")
        self.add_argument("distribution", type=lambda d: DistributionType[d.upper()], choices=list(DistributionType))
        self.add_argument("range", nargs=3)

    def process_args(self) -> None:
        if not self.dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file {self.dataset_file.name} does not exist!")

        if not self.field.is_file():
            raise FileNotFoundError(f"Field file {self.field.name} does not exist!")

        if not self.gt_file.is_file():
            raise FileNotFoundError(f"GT file {self.gt_file.name} does not exist!")


def main() -> None:
    args = ArgumentParser().parse_args()

    if not args.experiment_folder.is_dir():
        args.experiment_folder.mkdir(parents=True)

    field = Field.from_file(args.field)
    dataset = YoloDataset.from_dataset_file(args.dataset_file)
    gt = read_topcon_data(args.gt_file)

    rng = np.random.default_rng(args.seed)

    locations = get_objects_in_field(field, dataset, gt, args.range[1], args.distribution, rng=rng)
    locations_file = args.experiment_folder / (args.field.stem + "_object_locations.kmz")

    write_gps_locations_to_kml(locations_file, locations, kmz=True)

    for i in range(args.range[0], args.range[1] + args.range[2], args.range[2]):
        config_file = args.experiment_folder / f"executor_{i}_objects_{args.field.stem}.yaml"
        write_executor_config(config_file, args.gt_file, locations_file, i, seed=args.seed)


if __name__ == "__main__":
    main()

    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240213_clustered_1/trainings_dataset_12m/data.yml fields/clustered_1.yaml /home/abe/data/20240213_clustered_1/plants_clustered_1_RDNAPTRANS2008.csv clustered 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240423_clustered_2/trainings_dataset_12m/data.yml fields/clustered_2.yaml /home/abe/data/20240423_clustered_2/plants_clustered_2_RDNAPTRANS2008.csv clustered 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240718_clustered_3/trainings_dataset_12m/data.yml fields/clustered_3.yaml /home/abe/data/20240718_clustered_3/plants_clustered_3_RDNAPTRANS2008.csv clustered 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240801_clustered_4/trainings_dataset_12m/data.yml fields/clustered_4.yaml /home/abe/data/20240801_clustered_4/plants_clustered_4_RDNAPTRANS2008.csv clustered 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240213_uniform_1/trainings_dataset_12m/data.yml fields/uniform_1.yaml /home/abe/data/20240213_uniform_1/plants_uniform_1_RDNAPTRANS2008.csv uniform 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240423_uniform_2/trainings_dataset_12m/data.yml fields/uniform_2.yaml /home/abe/data/20240423_uniform_2/plants_uniform_2_RDNAPTRANS2008.csv uniform 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240718_uniform_3/trainings_dataset_12m/data.yml fields/uniform_3.yaml /home/abe/data/20240718_uniform_3/plants_uniform_3_RDNAPTRANS2008.csv uniform 0 200 20
    # create_number_of_objects_experiment experiments/number_of_objects /home/abe/data/20240801_uniform_4/trainings_dataset_12m/data.yml fields/uniform_4.yaml /home/abe/data/20240801_uniform_4/plants_uniform_4_RDNAPTRANS2008.csv uniform 0 200 20
