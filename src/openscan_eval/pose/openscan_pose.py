from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np


def angles_to_matrix(phi_deg: float, theta_deg: float) -> np.ndarray:
    """Object rotation Rz(theta) @ Rx(phi), right-handed, degrees."""
    p,t=map(math.radians,(phi_deg,theta_deg)); cp,sp,ct,st=math.cos(p),math.sin(p),math.cos(t),math.sin(t)
    rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]],float)
    rz=np.array([[ct,-st,0],[st,ct,0],[0,0,1]],float)
    return rz@rx


def metadata_records(frames):
    records=[]
    for frame in frames:
        item=asdict(frame); rotation=angles_to_matrix(frame.phi_deg,frame.theta_deg)
        transform=np.eye(4); transform[:3,:3]=rotation
        item.update({"rotation_matrix":rotation.tolist(),"transform_matrix":transform.tolist(),
                     "coordinate_convention":"right_handed_object_rotation_z_then_x"})
        records.append(item)
    return records
