from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

from ..dataset.loader import Dataset
from ..pose.openscan_pose import metadata_records


def export_pytorch3d(dataset: Dataset, output: Path, quality: list[dict]) -> Path:
    """Export only RGBA handoff images plus manifest and commanded-pose metadata."""
    target=output/"exports"/"pytorch3d"; rgba_target=target/"rgba";rgba_target.mkdir(parents=True,exist_ok=True)
    status={r["image"]:r["status"] for r in quality};rows=[];total=len(dataset.frames);interval=max(1,total//10)
    for index,frame in enumerate(dataset.frames,1):
        stem=Path(frame.image).stem;source=output/"processed"/"rgba"/(stem+".png")
        if not source.is_file(): raise FileNotFoundError(f"Processed RGBA missing: {source}")
        shutil.copy2(source,rgba_target/(stem+".png"))
        rows.append({"rgba":f"rgba/{stem}.png","phi_deg":frame.phi_deg,"theta_deg":frame.theta_deg,
                     "pose_source":"openscan_commanded","quality_status":status[frame.image],
                     "include":str(status[frame.image]!="REJECT").lower()})
        if index==1 or index%interval==0 or index==total: logging.info("导出数据：%d/%d",index,total)
    with (target/"dataset_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    metadata={"pose_source":"openscan_commanded","coordinate_convention":"right_handed_object_rotation_z_then_x",
              "warning":"Commanded mechanical positions are initialization metadata, not calibrated poses.",
              "frames":metadata_records(dataset.frames)}
    (target/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return target
