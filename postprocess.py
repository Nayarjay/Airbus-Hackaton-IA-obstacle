# postprocess.py
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_dbscan(points_xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Retourne labels DBSCAN (-1 = bruit)."""
    if len(points_xyz) == 0:
        return np.array([], dtype=np.int32)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xyz).astype(np.int32)

def bbox_aabb(points_xyz: np.ndarray):
    """
    BBox axis-aligned (simple).
    Retourne: center(x,y,z), size(width,length,height), yaw(0.0)
    """
    mn = points_xyz.min(axis=0)
    mx = points_xyz.max(axis=0)
    center = (mn + mx) / 2.0
    size = (mx - mn)
    yaw = 0.0
    return center, size, yaw
