import numpy as np
import torch
from torch.utils.data import Dataset
from configs import NUM_POINTS, RGB_TO_CLASS
from lidar_utils import open_lidar_dataset, iter_pose_ranges

def rgb_to_class_id(r, g, b):
    return RGB_TO_CLASS.get((int(r), int(g), int(b)), -1)

def spherical_to_xyz(arr):
    r = arr["distance_cm"].astype(np.float32) / 100.0
    az = np.deg2rad(arr["azimuth_raw"].astype(np.float32) / 100.0)
    el = np.deg2rad(arr["elevation_raw"].astype(np.float32) / 100.0)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1).astype(np.float32)

class LidarFrameDataset(Dataset):
    def __init__(self, h5_paths, train=True, num_points=NUM_POINTS):
        self.h5_paths = list(h5_paths)
        self.train = train
        self.num_points = int(num_points)
        self.frames = []  # list of (path, pose_dict, start, end)

        for p in self.h5_paths:
            f, ds = open_lidar_dataset(p)
            try:
                for pose, start, end in iter_pose_ranges(ds):
                    self.frames.append((p, pose, start, end))
            finally:
                f.close()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path, pose, start, end = self.frames[idx]

        f, ds = open_lidar_dataset(path)
        try:
            arr = ds[start:end]  # numpy structured array
        finally:
            f.close()

        xyz = spherical_to_xyz(arr)

        if self.train:
            y = np.array([rgb_to_class_id(r,g,b) for r,g,b in zip(arr["r"], arr["g"], arr["b"])], dtype=np.int64)
        else:
            y = np.full((len(xyz),), -1, dtype=np.int64)

        n = len(xyz)
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)

        xyz = xyz[choice]
        y = y[choice]

        return torch.from_numpy(xyz), torch.from_numpy(y), pose
