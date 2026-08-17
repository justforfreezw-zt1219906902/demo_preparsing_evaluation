import numpy as np
from scipy.spatial import cKDTree
import trimesh

def sampled_distances(reference,reconstruction,count=20000,seed=42):
    np.random.seed(seed)
    rp,_=trimesh.sample.sample_surface(reference,count); qp,_=trimesh.sample.sample_surface(reconstruction,count)
    r_to_q=cKDTree(qp).query(rp)[0]; q_to_r=cKDTree(rp).query(qp)[0]
    return rp,qp,r_to_q,q_to_r

def distance_metrics(reference,reconstruction,count=20000,seed=42):
    _,_,a,b=sampled_distances(reference,reconstruction,count,seed)
    combined=np.concatenate((a,b))
    return {"mean_surface_distance_mm":float(b.mean()),"p95_surface_distance_mm":float(np.percentile(b,95)),
            "chamfer_distance_mm2":float(np.mean(a*a)+np.mean(b*b)),"symmetric_mean_distance_mm":float(combined.mean()),
            "sample_count_per_mesh":count,"units":"mesh coordinate units; assumed millimetres"}
