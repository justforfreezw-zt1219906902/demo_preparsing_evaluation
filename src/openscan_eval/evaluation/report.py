from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import trimesh
from .alignment import rigid_icp
from .distances import distance_metrics
from .mesh_loader import load_mesh
from .visualization import heatmap_and_histogram,overlays

def compare_meshes(reference_path,reconstruction_path,output:Path,config:dict):
    cfg=config["evaluation"]; output.mkdir(parents=True,exist_ok=True); (output/"meshes").mkdir(exist_ok=True); (output/"views").mkdir(exist_ok=True)
    reference,reconstruction=load_mesh(reference_path),load_mesh(reconstruction_path); aligned=False; transform=np.eye(4)
    if cfg.get("rigid_alignment",False):
        n=min(10000,int(cfg["sample_count"])); np.random.seed(int(cfg["random_seed"]))
        reference_points,_=trimesh.sample.sample_surface(reference,n); reconstruction_points,_=trimesh.sample.sample_surface(reconstruction,n)
        transform=rigid_icp(reconstruction_points,reference_points); reconstruction.apply_transform(transform); aligned=True
    reconstruction.export(output/"meshes"/"aligned_reconstruction.ply")
    metrics=distance_metrics(reference,reconstruction,int(cfg["sample_count"]),int(cfg["random_seed"])); metrics.update({"alignment_applied":aligned,"scale_alignment_applied":False,"rigid_transform":transform.tolist(),"reference_mesh":str(reference_path),"reconstruction_mesh":str(reconstruction_path)})
    overlays(reference,reconstruction,output/"views"); heatmap_and_histogram(reference,reconstruction,output,float(cfg["heatmap_max_mm"]))
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    with (output/"metrics.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=metrics.keys()); w.writeheader(); w.writerow(metrics)
    images=''.join(f'<img src="views/{name}" style="max-width:48%">' for name in ["overlay_front.png","overlay_side.png","overlay_top.png","overlay_iso.png","distance_heatmap.png"])
    table=''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in metrics.items() if not isinstance(v,list))
    (output/"report.html").write_text(f'<!doctype html><meta charset="utf-8"><title>OpenScan evaluation</title><h1>Mesh comparison</h1><table>{table}</table>{images}<img src="distance_histogram.png" style="max-width:70%">')
    return metrics
