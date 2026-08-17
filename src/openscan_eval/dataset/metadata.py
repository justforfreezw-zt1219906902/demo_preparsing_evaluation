from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = ("image", "position_index", "phi_deg", "theta_deg")


@dataclass(frozen=True)
class FrameMetadata:
    image: str
    position_index: int
    phi_deg: float
    theta_deg: float
    focus_index: int | None = None
    focus_value: float | None = None
    pose_source: str = "openscan_commanded"


def _optional_number(value: str | None, convert):
    return None if value is None or not value.strip() else convert(value)


def read_positions_csv(path: str | Path) -> list[FrameMetadata]:
    """Parse OpenScan positions while preserving row-specific parse errors."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
        frames: list[FrameMetadata] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                image = (row["image"] or "").strip()
                if not image:
                    raise ValueError("image is empty")
                frames.append(
                    FrameMetadata(
                        image=image,
                        position_index=int(row["position_index"]),
                        phi_deg=float(row["phi_deg"]),
                        theta_deg=float(row["theta_deg"]),
                        focus_index=_optional_number(row.get("focus_index"), int),
                        focus_value=_optional_number(row.get("focus_value"), float),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CSV row {row_number}: {exc}") from exc
    return frames

