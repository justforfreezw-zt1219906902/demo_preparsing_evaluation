from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import cv2

from ..dataset.loader import Dataset
from .contrast import contrast_metric
from .exposure import exposure_metrics
from .sharpness import sharpness_metrics


def _classify(sharpness: float, contrast: float, dark: float, bright: float, cfg: dict[str, Any]):
    reasons = []
    status = "GOOD"
    if sharpness < cfg["reject_sharpness"]: status, reasons = "REJECT", ["blur"]
    elif sharpness < cfg["warning_sharpness"]: status, reasons = "WARNING", ["soft"]
    if contrast < cfg["reject_contrast"]: status, reasons = "REJECT", reasons + ["low_contrast"]
    elif contrast < cfg["warning_contrast"] and status == "GOOD": status, reasons = "WARNING", reasons + ["low_contrast"]
    clipping = max(dark, bright)
    if clipping >= cfg["reject_clip_percent"]: status, reasons = "REJECT", reasons + ["clipping"]
    elif clipping >= cfg["warning_clip_percent"] and status == "GOOD": status, reasons = "WARNING", reasons + ["clipping"]
    return status, ";".join(dict.fromkeys(reasons))


def _overrides(path: Path) -> dict[str, str]:
    if not path.is_file(): return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {r["image"].strip(): r["status"].strip().upper() for r in csv.DictReader(f)}


def analyze_quality(dataset: Dataset, config: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    cfg = config["quality"]
    overrides = _overrides(dataset.root / cfg["override_file"])
    rows = []
    total=len(dataset.frames); interval=max(1,total//10)
    for index,frame in enumerate(dataset.frames,1):
        image = cv2.imread(str(dataset.image_path(frame)))
        h, w = image.shape[:2]
        scale = min(1.0, cfg["analysis_max_dimension"] / max(h, w))
        if scale < 1: image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Central ROI reduces domination by the fixed wall/background.
        roi = gray[int(gray.shape[0]*.12):int(gray.shape[0]*.92), int(gray.shape[1]*.18):int(gray.shape[1]*.82)]
        lap, tenengrad = sharpness_metrics(roi)
        mean, dark, bright = exposure_metrics(roi, cfg["dark_clip_threshold"], cfg["bright_clip_threshold"])
        contrast = contrast_metric(roi)
        status, reason = _classify(lap, contrast, dark, bright, cfg)
        if frame.image in overrides:
            status, reason = overrides[frame.image], "manual_override"
        rows.append({"image": frame.image, "phi_deg": frame.phi_deg, "theta_deg": frame.theta_deg,
                     "sharpness": round(lap,3), "tenengrad": round(tenengrad,3), "brightness": round(mean,3),
                     "dark_clip_pct": round(dark,3), "bright_clip_pct": round(bright,3),
                     "contrast": round(contrast,3), "status": status, "reason": reason})
        if index==1 or index%interval==0 or index==total: logging.info("质量分析：%d/%d",index,total)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "quality_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
        writer.writeheader(); writer.writerows(rows)
    return rows
