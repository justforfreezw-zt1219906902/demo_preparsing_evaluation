from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .metadata import FrameMetadata, read_positions_csv


@dataclass(frozen=True)
class Dataset:
    root: Path
    images_dir: Path
    positions_csv: Path
    frames: tuple[FrameMetadata, ...]

    def image_path(self, frame: FrameMetadata) -> Path:
        return self.images_dir / frame.image


def load_dataset(config: dict[str, Any], root: str | Path | None = None) -> Dataset:
    """Load a flat dataset, taking its directory from the configured env var by default."""
    settings = config["dataset"]
    env_var = settings.get("env_var", "OPENSCAN_DATASET_DIR")
    location = root if root is not None else os.environ.get(env_var)
    if not location:
        raise ValueError(f"Dataset directory is not set. Define {env_var} or pass --dataset.")
    root_path = Path(location).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")
    csv_path = root_path / settings["positions_csv"]
    if not csv_path.is_file():
        raise FileNotFoundError(f"Positions CSV not found: {csv_path}")
    frames = tuple(read_positions_csv(csv_path))
    return Dataset(
        root=root_path,
        images_dir=root_path / settings["images_dir"],
        positions_csv=csv_path,
        frames=frames,
    )
