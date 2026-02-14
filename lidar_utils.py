import numpy as np
import h5py

POSE_FIELDS = ("ego_x", "ego_y", "ego_z", "ego_yaw")

def iter_pose_ranges(ds, chunk_size=2_000_000):
    """
    Yield: (pose_dict, start, end) pour chaque bloc contigu de même pose.
    """
    n = ds.shape[0]
    start = 0

    # Lire la première pose
    first = ds[0]
    current_pose = tuple(float(first[f]) for f in POSE_FIELDS)

    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        chunk = ds[i:j]

        poses = np.vstack([chunk[f].astype(np.float32) for f in POSE_FIELDS]).T  # (M,4)

        # repérer où la pose change
        diffs = np.any(poses[1:] != poses[:-1], axis=1)
        change_idx = np.where(diffs)[0]

        if change_idx.size == 0:
            i = j
            continue

        # traiter les changements un par un (dans ce chunk)
        for rel in change_idx:
            # rel = index dans (poses[1:] vs poses[:-1]) donc coupure à i+rel+1
            cut = i + rel + 1

            pose_dict = {k: current_pose[t] for t, k in enumerate(POSE_FIELDS)}
            yield pose_dict, start, cut

            # nouvelle pose
            nxt = ds[cut]
            current_pose = tuple(float(nxt[f]) for f in POSE_FIELDS)
            start = cut

        i = j

    # dernier bloc
    pose_dict = {k: current_pose[t] for t, k in enumerate(POSE_FIELDS)}
    yield pose_dict, start, n

def open_lidar_dataset(h5_path: str):
    f = h5py.File(h5_path, "r")
    # Support underscore OU dash
    if "lidar_points" in f:
        return f, f["lidar_points"]
    if "lidar-points" in f:
        return f, f["lidar-points"]
    raise ValueError(f"Aucun dataset lidar_points / lidar-points trouvé dans {h5_path}. Clés: {list(f.keys())}")

