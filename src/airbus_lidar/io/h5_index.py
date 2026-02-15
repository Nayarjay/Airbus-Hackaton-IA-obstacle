from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import os

import h5py
import numpy as np


@dataclass(frozen=True)
class FrameMeta:
    file_path: str
    frame_index: int
    start: int
    end: int
    ego_x: int
    ego_y: int
    ego_z: int
    ego_yaw: int

    @property
    def pose_tuple(self) -> Tuple[int, int, int, int]:
        return (self.ego_x, self.ego_y, self.ego_z, self.ego_yaw)

    @property
    def num_points_raw(self) -> int:
        return int(self.end - self.start)


class H5FrameIndex:
    """
    Hypothèse (cohérente avec "concatenated frames"):
    les points d'une frame sont contigus et la pose ego change entre frames.
    On construit alors un index (start,end) par frame en O(N) sans pandas.
    """

    def __init__(self, file_path: str, dataset_name: str = "lidar_points"):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.frames: List[FrameMeta] = []

    def build(self) -> "H5FrameIndex":
        with h5py.File(self.file_path, "r") as f:
            if self.dataset_name not in f:
                raise ValueError(f"Dataset '{self.dataset_name}' not found in {self.file_path}")
            dset = f[self.dataset_name]

            ego_x = dset["ego_x"][:]
            ego_y = dset["ego_y"][:]
            ego_z = dset["ego_z"][:]
            ego_yaw = dset["ego_yaw"][:]

        # changements de pose => start d'une nouvelle frame
        change = np.zeros(len(ego_x), dtype=bool)
        change[0] = True
        change[1:] = (ego_x[1:] != ego_x[:-1]) | (ego_y[1:] != ego_y[:-1]) | (ego_z[1:] != ego_z[:-1]) | (ego_yaw[1:] != ego_yaw[:-1])
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], len(ego_x))

        frames = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            frames.append(
                FrameMeta(
                    file_path=self.file_path,
                    frame_index=i,
                    start=int(s),
                    end=int(e),
                    ego_x=int(ego_x[s]),
                    ego_y=int(ego_y[s]),
                    ego_z=int(ego_z[s]),
                    ego_yaw=int(ego_yaw[s]),
                )
            )

        self.frames = frames
        return self

    def save(self, cache_path: str) -> None:
        os.makedirs(str(Path(cache_path).parent), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(cache_path: str) -> "H5FrameIndex":
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, H5FrameIndex):
            raise TypeError("Cache file does not contain a H5FrameIndex")
        return obj
