from __future__ import annotations
import csv,json
from pathlib import Path

def create_dataset_report(output:Path):
    quality=list(csv.DictReader((output/"quality_report.csv").open()))
    counts={s:sum(r["status"]==s for r in quality) for s in ("GOOD","WARNING","REJECT")}
    metrics={}
    if (output/"evaluation"/"metrics.json").is_file(): metrics=json.loads((output/"evaluation"/"metrics.json").read_text())
    rows=''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in {"total_images":len(quality),**counts,**{k:v for k,v in metrics.items() if not isinstance(v,list)}}.items())
    (output/"report.html").write_text(f'<!doctype html><meta charset="utf-8"><title>OpenScan report</title><h1>OpenScan processing report</h1><table>{rows}</table><p>Commanded poses are initialization metadata, not calibrated camera poses.</p>')

def summarize(root:Path):
    rows=[]
    for metrics_path in root.rglob("evaluation/metrics.json"):
        output=metrics_path.parent.parent; quality_path=output/"quality_report.csv"
        if not quality_path.is_file(): continue
        quality=list(csv.DictReader(quality_path.open())); metrics=json.loads(metrics_path.read_text()); sharp=sorted(float(r["sharpness"]) for r in quality)
        rows.append({"dataset":str(output),"total_images":len(quality),"good_images":sum(r["status"]=="GOOD" for r in quality),"warning_images":sum(r["status"]=="WARNING" for r in quality),"rejected_images":sum(r["status"]=="REJECT" for r in quality),"median_sharpness":sharp[len(sharp)//2],"mean_surface_distance_mm":metrics["mean_surface_distance_mm"],"p95_surface_distance_mm":metrics["p95_surface_distance_mm"],"chamfer_distance_mm2":metrics["chamfer_distance_mm2"]})
    target=root/"experiment_summary.csv"
    if rows:
        with target.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return target,len(rows)
