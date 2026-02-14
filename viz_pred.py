import argparse
import numpy as np
import pandas as pd
import open3d as o3d
import h5py


POSE_FIELDS = ("ego_x", "ego_y", "ego_z", "ego_yaw")


def open_lidar_dataset(h5_path: str):
    f = h5py.File(h5_path, "r")
    if "lidar_points" in f:
        return f, f["lidar_points"]
    if "lidar-points" in f:
        return f, f["lidar-points"]
    keys = list(f.keys())
    f.close()
    raise ValueError(f"Aucun dataset lidar_points / lidar-points trouvé dans {h5_path}. Clés: {keys}")


def iter_pose_ranges(ds, chunk_size=2_000_000):
    """
    Yield (pose_dict, start, end) pour chaque bloc contigu de même pose.
    Hypothèse: les points sont stockés par pose en blocs.
    """
    n = ds.shape[0]
    if n == 0:
        return

    first = ds[0]
    current_pose = tuple(float(first[f]) for f in POSE_FIELDS)

    start = 0
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        chunk = ds[i:j]

        poses = np.vstack([chunk[f].astype(np.float32) for f in POSE_FIELDS]).T  # (M,4)
        diffs = np.any(poses[1:] != poses[:-1], axis=1)
        change_idx = np.where(diffs)[0]

        if change_idx.size == 0:
            i = j
            continue

        for rel in change_idx:
            cut = i + rel + 1
            pose_dict = {k: current_pose[t] for t, k in enumerate(POSE_FIELDS)}
            yield pose_dict, start, cut

            nxt = ds[cut]
            current_pose = tuple(float(nxt[f]) for f in POSE_FIELDS)
            start = cut

        i = j

    pose_dict = {k: current_pose[t] for t, k in enumerate(POSE_FIELDS)}
    yield pose_dict, start, n


def spherical_to_xyz(arr):
    """
    Convertit (distance_cm, azimuth_raw, elevation_raw) -> (x,y,z) en mètres.
    azimuth/elevation sont en centi-degrés (1/100 deg).
    """
    r = arr["distance_cm"].astype(np.float32) / 100.0
    az = np.deg2rad(arr["azimuth_raw"].astype(np.float32) / 100.0)
    el = np.deg2rad(arr["elevation_raw"].astype(np.float32) / 100.0)

    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def local_to_world(xyz: np.ndarray, pose: dict) -> np.ndarray:
    """
    xyz en mètres repère local LiDAR
    ego_x,y,z en cm ; ego_yaw en centi-degrés
    """
    ego_x_m = float(pose["ego_x"]) / 100.0
    ego_y_m = float(pose["ego_y"]) / 100.0
    ego_z_m = float(pose["ego_z"]) / 100.0
    yaw_rad = np.deg2rad(float(pose["ego_yaw"]) / 100.0)

    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]], dtype=np.float32)

    return (xyz @ R.T) + np.array([ego_x_m, ego_y_m, ego_z_m], dtype=np.float32)


def make_bbox(row: pd.Series):
    """
    Open3D OrientedBoundingBox(center, R, extent)
    yaw = bbox_yaw (rad) (si ton CSV est en rad). Si ton yaw est en degrés, convertis ici.
    """
    center = np.array([row["bbox_center_x"], row["bbox_center_y"], row["bbox_center_z"]], dtype=np.float64)
    extent = np.array([row["bbox_width"], row["bbox_length"], row["bbox_height"]], dtype=np.float64)
    yaw = float(row["bbox_yaw"])

    R = o3d.geometry.get_rotation_matrix_from_xyz([0.0, 0.0, yaw])
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return obb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="H5 (train/eval)")
    ap.add_argument("--csv", required=True, help="CSV de prédiction (bboxes)")
    ap.add_argument("--frame", type=int, default=0, help="index de pose/frame à afficher")
    ap.add_argument("--max_points", type=int, default=200000, help="downsample pour affichage")
    ap.add_argument("--world", action="store_true", help="afficher en repère monde (recommandé)")
    args = ap.parse_args()

    preds = pd.read_csv(args.csv)

    f, ds = open_lidar_dataset(args.file)
    try:
        frames = list(iter_pose_ranges(ds))
        if len(frames) == 0:
            raise SystemExit("Aucune frame/pose détectée dans ce fichier.")

        if args.frame < 0 or args.frame >= len(frames):
            raise SystemExit(f"--frame invalide. Range: 0..{len(frames)-1}")

        pose, start, end = frames[args.frame]
        arr = ds[start:end]

        # filtre points invalides
        if "distance_cm" in arr.dtype.names:
            arr = arr[arr["distance_cm"] > 0]

        xyz = spherical_to_xyz(arr)

        if args.world:
            xyz = local_to_world(xyz, pose)

        # downsample pour visu
        if len(xyz) > args.max_points:
            idx = np.random.choice(len(xyz), args.max_points, replace=False)
            xyz = xyz[idx]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # couleurs: si le H5 a r,g,b on les affiche
        if all(n in arr.dtype.names for n in ("r", "g", "b")):
            rgb = np.stack([arr["r"], arr["g"], arr["b"]], axis=1).astype(np.float32) / 255.0
            if len(rgb) > args.max_points:
                rgb = rgb[idx]
            pcd.colors = o3d.utility.Vector3dVector(rgb)

        # filtre bboxes correspondant à la pose
        # (attention: exact float match. si ça rate, on fera une comparaison tolérante)
        m = (
            (preds["ego_x"] == pose["ego_x"]) &
            (preds["ego_y"] == pose["ego_y"]) &
            (preds["ego_z"] == pose["ego_z"]) &
            (preds["ego_yaw"] == pose["ego_yaw"])
        )

        bboxes = []
        for _, row in preds[m].iterrows():
            obb = make_bbox(row)
            obb.color = (1.0, 0.0, 0.0)
            bboxes.append(obb)

        print("Frame:", args.frame, "Pose:", pose)
        print("Points:", len(xyz), "BBoxes:", len(bboxes))
        o3d.visualization.draw_geometries([pcd] + bboxes)
    finally:
        f.close()


if __name__ == "__main__":
    main()
