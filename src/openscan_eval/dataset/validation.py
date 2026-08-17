from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from PIL import Image, UnidentifiedImageError

from .loader import Dataset


def validate_dataset(dataset: Dataset, config: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    dimensions: dict[str, list[int]] = {}
    names = [frame.image for frame in dataset.frames]
    for name, count in Counter(names).items():
        if count > 1:
            issues.append({"severity": "error", "code": "duplicate_image", "image": name, "count": count})

    indices = [frame.position_index for frame in dataset.frames]
    for index, count in Counter(indices).items():
        if index < 0:
            issues.append({"severity": "error", "code": "invalid_position_index", "position_index": index})
        if count > 1:
            issues.append({"severity": "error", "code": "duplicate_position_index", "position_index": index})

    for frame in dataset.frames:
        image_path = dataset.image_path(frame)
        if not image_path.is_file():
            issues.append({"severity": "error", "code": "missing_image", "image": frame.image})
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                dimensions[frame.image] = [image.width, image.height]
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            issues.append({"severity": "error", "code": "corrupted_image", "image": frame.image, "detail": str(exc)})

    unique_dimensions = {tuple(value) for value in dimensions.values()}
    if config["dataset"].get("require_consistent_dimensions", True) and len(unique_dimensions) > 1:
        issues.append({"severity": "error", "code": "inconsistent_dimensions", "dimensions": sorted([list(v) for v in unique_dimensions])})

    return {
        "schema_version": 1,
        "dataset": str(dataset.root),
        "positions_csv": str(dataset.positions_csv),
        "images_dir": str(dataset.images_dir),
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "frame_count": len(dataset.frames),
        "decoded_image_count": len(dimensions),
        "frames": [asdict(frame) for frame in dataset.frames],
        "image_dimensions": dimensions,
        "issues": issues,
    }
