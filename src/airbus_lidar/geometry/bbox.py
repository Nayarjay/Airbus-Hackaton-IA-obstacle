from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from src.airbus_lidar.constants import CLASS_ID_TO_LABEL
from src.airbus_lidar.config import ClusterConfig


@dataclass
class BBox3D:
    cx: float
    cy: float
    cz: float
    width: float
    length: float
    height: float
    yaw: float
    class_id: int

    @property
    def class_label(self) -> str:
        return CLASS_ID_TO_LABEL[self.class_id]


def _pca_obb_xy(points_xyz: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    PCA 2D sur XY pour yaw + OBB.
    Retourne (yaw, center_xyz, dims_wlh) avec dims=(width,length,height) avant rotation yaw.
    """
    xy = points_xyz[:, :2].astype(np.float32)
    mean_xy = xy.mean(axis=0)

    X = xy - mean_xy[None, :]
    cov = (X.T @ X) / max(len(X) - 1, 1)

    # cas dégénéré (très peu de variance)
    if not np.isfinite(cov).all():
        yaw = 0.0
    else:
        eigvals, eigvecs = np.linalg.eigh(cov)  # tri croissant
        v = eigvecs[:, 1]
        yaw = float(np.arctan2(v[1], v[0]))

    c = float(np.cos(-yaw))
    s = float(np.sin(-yaw))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)

    xy_rot = (X @ R.T)
    min_xy = xy_rot.min(axis=0)
    max_xy = xy_rot.max(axis=0)

    width = float(max_xy[0] - min_xy[0])
    length = float(max_xy[1] - min_xy[1])

    z = points_xyz[:, 2].astype(np.float32)
    min_z = float(z.min())
    max_z = float(z.max())
    height = float(max_z - min_z)

    center_rot = 0.5 * (min_xy + max_xy)

    c2 = float(np.cos(yaw))
    s2 = float(np.sin(yaw))
    R2 = np.array([[c2, -s2],
                   [s2,  c2]], dtype=np.float32)
    center_xy = (center_rot @ R2.T) + mean_xy

    cx, cy = float(center_xy[0]), float(center_xy[1])
    cz = float(0.5 * (min_z + max_z))

    dims = np.array([width, length, height], dtype=np.float32)
    center = np.array([cx, cy, cz], dtype=np.float32)
    return yaw, center, dims


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Garde 1 point par voxel (rapide, numpy)."""
    if voxel <= 0.0 or len(points) == 0:
        return points
    coords = np.floor(points / np.float32(voxel)).astype(np.int32)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return points[idx]


def _limit_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), size=int(max_points), replace=False)
    return points[idx]


def cluster_and_build_bboxes(
    xyz: np.ndarray,
    pred_class: np.ndarray,
    cluster_cfg: ClusterConfig,
) -> List[BBox3D]:
    """
    xyz: (N,3) mètres
    pred_class: (N,) classes 0..3 (sans background)
    """
    out: List[BBox3D] = []

    for cid in [0, 1, 2, 3]:
        mask = (pred_class == cid)
        pts = xyz[mask]
        if len(pts) < cluster_cfg.min_cluster_points:
            continue

        eps = float(cluster_cfg.eps_by_class.get(cid, 1.0))
        min_s = int(cluster_cfg.min_samples_by_class.get(cid, 10))

        voxel = float(getattr(cluster_cfg, "voxel_size_by_class", {}).get(cid, 0.0))
        maxp = int(getattr(cluster_cfg, "max_points_by_class", {}).get(cid, 50000))

        # downsample (préserve mieux la géométrie que le random pur)
        pts_ds = _voxel_downsample(pts, voxel)
        pts_ds = _limit_points(pts_ds, maxp)
        if len(pts_ds) < cluster_cfg.min_cluster_points:
            continue

        labels = DBSCAN(eps=eps, min_samples=min_s, n_jobs=-1).fit_predict(pts_ds)

        for lab in np.unique(labels):
            if lab == -1:
                continue

            cluster_pts = pts_ds[labels == lab]
            if len(cluster_pts) < cluster_cfg.min_cluster_points:
                continue

            yaw, center, dims = _pca_obb_xy(cluster_pts)

            # marge bbox pour englober davantage
            inflate = float(getattr(cluster_cfg, "inflate_by_class", {}).get(cid, 1.0))
            dims = dims * np.float32(inflate)

            # dimensions minimales (utile surtout câble)
            min_ext = getattr(cluster_cfg, "min_extent_by_class", {}).get(cid, None)
            if min_ext is not None:
                dims = np.maximum(dims, np.array(min_ext, dtype=np.float32))

            # sécurité contre dims nulles
            dims = np.maximum(dims, np.array([0.05, 0.05, 0.05], dtype=np.float32))

            out.append(
                BBox3D(
                    cx=float(center[0]),
                    cy=float(center[1]),
                    cz=float(center[2]),
                    width=float(dims[0]),
                    length=float(dims[1]),
                    height=float(dims[2]),
                    yaw=float(yaw),
                    class_id=cid,
                )
            )

    return out
