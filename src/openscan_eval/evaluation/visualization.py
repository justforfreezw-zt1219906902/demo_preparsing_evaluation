from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

def _points(mesh,n=12000):
    if len(mesh.vertices)<=n:return np.asarray(mesh.vertices)
    return np.asarray(mesh.vertices)[np.linspace(0,len(mesh.vertices)-1,n,dtype=int)]

def overlays(reference,reconstruction,out:Path):
    out.mkdir(parents=True,exist_ok=True); r,q=_points(reference),_points(reconstruction)
    views={"front":(0,2),"side":(1,2),"top":(0,1),"iso":None}
    for name,axes in views.items():
        fig=plt.figure(figsize=(6,6))
        if axes:
            ax=fig.add_subplot(); ax.scatter(r[:,axes[0]],r[:,axes[1]],s=.2,c="royalblue",alpha=.35,label="reference")
            ax.scatter(q[:,axes[0]],q[:,axes[1]],s=.2,c="orangered",alpha=.35,label="reconstruction"); ax.set_aspect("equal")
        else:
            ax=fig.add_subplot(projection="3d"); ax.scatter(*r.T,s=.15,c="royalblue",alpha=.3); ax.scatter(*q.T,s=.15,c="orangered",alpha=.3)
        ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(out/f"overlay_{name}.png",dpi=160); plt.close(fig)

def heatmap_and_histogram(reference,reconstruction,out:Path,max_mm=5.0):
    vertices=np.asarray(reconstruction.vertices); distances=cKDTree(np.asarray(reference.vertices)).query(vertices)[0]
    colors=plt.get_cmap("turbo")(np.clip(distances/max_mm,0,1)); mesh=reconstruction.copy(); mesh.visual.vertex_colors=(colors*255).astype(np.uint8)
    mesh.export(out/"meshes"/"distance_heatmap.ply")
    fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(projection="3d"); p=ax.scatter(*vertices.T,c=distances,s=.5,cmap="turbo",vmin=0,vmax=max_mm); fig.colorbar(p,label="distance (mm)"); fig.tight_layout(); fig.savefig(out/"views"/"distance_heatmap.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist(distances,bins=60,range=(0,max(max_mm,float(np.percentile(distances,99)))),color="steelblue"); ax.set(xlabel="distance (mm)",ylabel="vertices"); fig.tight_layout(); fig.savefig(out/"distance_histogram.png",dpi=160); plt.close(fig)
    return distances
