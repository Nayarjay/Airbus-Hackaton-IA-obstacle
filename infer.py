# infer.py
import argparse
import os
import numpy as np
import pandas as pd
import torch

from pointnet_seg import PointNetSeg
from configs import NUM_CLASSES, NUM_POINTS, CLASS_NAMES, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
from lidar_utils import open_lidar_dataset, iter_pose_ranges

from sklearn.cluster import DBSCAN

POSE_FIELDS = ("ego_x", "ego_y", "ego_z", "ego_yaw")

def spherical_to_xyz(arr):
    r = arr["distance_cm"].astype(np.float32) / 100.0
    az = np.deg2rad(arr["azimuth_raw"].astype(np.float32) / 100.0)
    el = np.deg2rad(arr["elevation_raw"].astype(np.float32) / 100.0)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def bbox_aabb(points_xyz: np.ndarray):
    mn = points_xyz.min(axis=0)
    mx = points_xyz.max(axis=0)
    center = (mn + mx) / 2.0
    size = (mx - mn)
    yaw = 0.0
    return center, size, yaw

def cluster_dbscan(points_xyz, eps, min_samples):
    if len(points_xyz) == 0:
        return np.array([], dtype=np.int32)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xyz).astype(np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--density", type=float, default=1.0, help="si tu veux re-downsample (sinon 1.0)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    model = PointNetSeg(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    f, ds = open_lidar_dataset(args.file)
    rows = []

    try:
        for pose, start, end in iter_pose_ranges(ds):
            arr = ds[start:end]

            # density downsample optionnel
            n = arr.shape[0]
            k = max(1, int(n * args.density))
            if k < n:
                idx = np.random.choice(n, k, replace=False)
                arr = arr[idx]

            xyz = spherical_to_xyz(arr)

            # sample NUM_POINTS pour PointNet
            n2 = len(xyz)
            if n2 >= NUM_POINTS:
                choice = np.random.choice(n2, NUM_POINTS, replace=False)
            else:
                choice = np.random.choice(n2, NUM_POINTS, replace=True)
            xyz_s = xyz[choice]

            with torch.no_grad():
                inp = torch.from_numpy(xyz_s).unsqueeze(0).to(device)  # (1,N,3)
                logits = model(inp)[0].cpu().numpy()                    # (N,C)
                pred = logits.argmax(axis=1).astype(np.int32)

            # post-process par classe
            for cid in range(NUM_CLASSES):
                pts = xyz_s[pred == cid]
                if len(pts) < 30:
                    continue

                labels = cluster_dbscan(
                    pts,
                    eps=float(DBSCAN_EPS.get(cid, 0.8)),
                    min_samples=int(DBSCAN_MIN_SAMPLES.get(cid, 10)),
                )

                for cl in np.unique(labels):
                    if cl == -1:
                        continue
                    blob = pts[labels == cl]
                    if len(blob) < 30:
                        continue

                    center, size, yaw = bbox_aabb(blob)

                    rows.append({
                        "ego_x": float(pose["ego_x"]),
                        "ego_y": float(pose["ego_y"]),
                        "ego_z": float(pose["ego_z"]),
                        "ego_yaw": float(pose["ego_yaw"]),
                        "bbox_center_x": float(center[0]),
                        "bbox_center_y": float(center[1]),
                        "bbox_center_z": float(center[2]),
                        "bbox_width": float(size[0]),
                        "bbox_length": float(size[1]),
                        "bbox_height": float(size[2]),
                        "bbox_yaw": float(yaw),
                        "Class ID": int(cid),
                        "Class Label": CLASS_NAMES[cid],
                    })
    finally:
        f.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
