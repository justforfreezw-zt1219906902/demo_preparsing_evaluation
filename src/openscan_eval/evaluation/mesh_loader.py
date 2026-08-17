from pathlib import Path
import trimesh

def load_mesh(path: str | Path):
    mesh=trimesh.load_mesh(path,force="mesh")
    if not isinstance(mesh,trimesh.Trimesh) or mesh.is_empty: raise ValueError(f"Not a usable mesh: {path}")
    return mesh
