from __future__ import annotations

from pathlib import Path

IMG_FILE_TYPES = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG", ".tif", ".TIF"]


def get_image_paths(image_folder: Path | list[Path]) -> list[Path]:
    img_paths = []

    if isinstance(image_folder, Path):
        image_folder = [image_folder]

    for img_folder in image_folder:
        for ext in IMG_FILE_TYPES:
            img_paths.extend(list(img_folder.glob(f"**/*{ext}")))

    return sorted(img_paths)
