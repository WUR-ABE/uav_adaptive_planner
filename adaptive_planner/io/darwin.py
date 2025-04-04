from __future__ import annotations

from json import load as json_load
from typing import TYPE_CHECKING
from zipfile import ZipFile

import numpy as np

from adaptive_planner.utils import Annotation

if TYPE_CHECKING:
    from pathlib import Path


def load_darwin_annotations(darwin_file: Path) -> tuple[dict[str, list[Annotation]], list[str]]:
    darwin_data: dict[str, list[Annotation]] = {}
    class_names = []

    with ZipFile(darwin_file) as zfile:
        for fname in zfile.namelist():
            tags = []

            with zfile.open(fname) as annotation_file:
                if annotation_file.name.startswith(".v7/"):
                    continue

                json_data = json_load(annotation_file)
                image_name = json_data["item"]["name"].split(".")[0]

                image_size = None
                for slot in json_data["item"]["slots"]:
                    if slot["type"] == "image":
                        image_size = np.array([slot["width"], slot["height"]], dtype=np.uint16)
                        break

                if image_size is None:
                    raise ValueError("Image size not in annotation file!")

                darwin_data[image_name] = []

                for bbox in json_data["annotations"]:
                    # If it has not a bounding box attribute, it's probably a tag
                    if "bounding_box" not in bbox:
                        tags.append(bbox["name"])
                        continue

                    class_names.append(bbox["name"])
                    darwin_data[image_name].append(
                        Annotation(
                            bbox["id"],
                            bbox["bounding_box"]["x"],
                            bbox["bounding_box"]["y"],
                            bbox["bounding_box"]["w"],
                            bbox["bounding_box"]["h"],
                            bbox["name"],
                            image_name,
                            image_size,
                        )
                    )

                # Apply tags to all annotations in this file
                for annotation in darwin_data[image_name]:
                    annotation.tags.extend(tags)

    return darwin_data, list(set(class_names))
