import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from openscan_eval.config import load_config
from openscan_eval.evaluation.alignment import rigid_transform
from openscan_eval.evaluation.distances import distance_metrics
from openscan_eval.pose.openscan_pose import angles_to_matrix
from openscan_eval.reporting import create_dataset_report


def test_angle_metadata_conversion():
    assert np.allclose(angles_to_matrix(0,90),[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-7)


def test_rigid_alignment():
    source=np.array([[0,0,0],[1,0,0],[0,1,0]],float); target=source+np.array([2,3,4])
    transform=rigid_transform(source,target)
    assert np.allclose(source@transform[:3,:3].T+transform[:3,3],target)


def test_synthetic_mesh_distance():
    mesh=trimesh.creation.box(extents=[10,10,10]); metrics=distance_metrics(mesh,mesh.copy(),3000,1)
    assert metrics["mean_surface_distance_mm"] < .8


def test_evaluation_report_generation(tmp_path):
    with (tmp_path/"quality_report.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["image","status","sharpness"]); w.writeheader(); w.writerow({"image":"a.jpg","status":"GOOD","sharpness":100})
    evaluation=tmp_path/"evaluation"; evaluation.mkdir(); (evaluation/"metrics.json").write_text(json.dumps({"mean_surface_distance_mm":1.0}))
    create_dataset_report(tmp_path)
    assert (tmp_path/"report.html").is_file()
