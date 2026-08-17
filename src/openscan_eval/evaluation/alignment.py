import numpy as np
from scipy.spatial import cKDTree

def rigid_transform(source: np.ndarray, target: np.ndarray):
    cs,ct=source.mean(0),target.mean(0); h=(source-cs).T@(target-ct)
    u,_,vt=np.linalg.svd(h); r=vt.T@u.T
    if np.linalg.det(r)<0: vt[-1]*=-1; r=vt.T@u.T
    t=ct-r@cs; transform=np.eye(4); transform[:3,:3]=r; transform[:3,3]=t
    return transform

def rigid_icp(source: np.ndarray,target: np.ndarray,iterations=30):
    moving=source.copy(); total=np.eye(4); tree=cKDTree(target)
    for _ in range(iterations):
        _,idx=tree.query(moving); step=rigid_transform(moving,target[idx]); moving=moving@step[:3,:3].T+step[:3,3]
        total=step@total
    return total
