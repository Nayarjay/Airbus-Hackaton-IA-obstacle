from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.airbus_lidar.config import DataConfig
from src.airbus_lidar.constants import RGB_TO_CLASS_ID, BACKGROUND_ID, NUM_CLASSES
from src.airbus_lidar.io.h5_index import H5FrameIndex, FrameMeta
from src.airbus_lidar.io.h5_reader import read_frame_fields
from src.airbus_lidar.geometry.coords import spherical_to_local_cartesian_np
from src.airbus_lidar.data.transforms import sample_or_pad_indices, TrainAugment


def rgb_to_class_id(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(r), BACKGROUND_ID, dtype=np.int64)
    for (R, G, B), cid in RGB_TO_CLASS_ID.items():
        m = (r == R) & (g == G) & (b == B)
        out[m] = cid
    return out


@dataclass
class Sample:
    x: torch.Tensor          # (C, N)
    y: torch.Tensor          # (N,)
    pose: Dict[str, int]     # ego_x,y,z,yaw (cm, 1/100 deg)


class LidarFrameDataset(Dataset):
    def __init__(
        self,
        frames: List[FrameMeta],
        data_cfg: DataConfig,
        train: bool,
        augment: Optional[TrainAugment] = None,
    ):
        self.frames = frames
        self.cfg = data_cfg
        self.train = train
        self.augment = augment if (train and augment is not None) else None

        self.fields = ["distance_cm", "azimuth_raw", "elevation_raw", "reflectivity", "r", "g", "b",
                       "ego_x", "ego_y", "ego_z", "ego_yaw"]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Sample:
        fm = self.frames[idx]
        d = read_frame_fields(
            file_path=fm.file_path,
            start=fm.start,
            end=fm.end,
            dataset_name=self.cfg.dataset_name,
            fields=self.fields,
        )

        # filtre points valides
        valid = d["distance_cm"] > 0
        if not np.any(valid):
            # frame vide => on renvoie du padding background
            N = self.cfg.num_points_train
            x = torch.zeros((4 if self.cfg.use_intensity else 3, N), dtype=torch.float32)
            y = torch.full((N,), BACKGROUND_ID, dtype=torch.long)
            pose = {"ego_x": fm.ego_x, "ego_y": fm.ego_y, "ego_z": fm.ego_z, "ego_yaw": fm.ego_yaw}
            return Sample(x=x, y=y, pose=pose)

        dist = d["distance_cm"][valid]
        az = d["azimuth_raw"][valid]
        el = d["elevation_raw"][valid]
        xyz = spherical_to_local_cartesian_np(dist, az, el)  # (n,3)

        if self.augment is not None:
            xyz = self.augment(xyz)

        # features
        if self.cfg.use_intensity:
            inten = d["reflectivity"][valid].astype(np.float32) / 255.0
            feats = np.concatenate([xyz, inten[:, None]], axis=1)  # (n,4)
        else:
            feats = xyz  # (n,3)

        # labels
        y_np = rgb_to_class_id(d["r"][valid], d["g"][valid], d["b"][valid])  # (n,)

        # sample fixed N (balanced obstacle/background)
        N = self.cfg.num_points_train
        obs_idx = np.where(y_np != BACKGROUND_ID)[0]
        bg_idx = np.where(y_np == BACKGROUND_ID)[0]

        n_obs = min(len(obs_idx), int(0.6 * N))  # 60% obstacles si possible
        n_bg = N - n_obs

        if n_obs > 0:
            pick_obs = np.random.choice(obs_idx, size=n_obs, replace=(len(obs_idx) < n_obs))
            if len(bg_idx) > 0:
                pick_bg = np.random.choice(bg_idx, size=n_bg, replace=(len(bg_idx) < n_bg))
            else:
                pick_bg = np.random.choice(obs_idx, size=n_bg, replace=True)
            inds = np.concatenate([pick_obs, pick_bg])
        else:
            inds = sample_or_pad_indices(len(feats), N)

        feats = feats[inds]
        y_np = y_np[inds]

        x = torch.from_numpy(feats.T).float()     # (C,N)
        y = torch.from_numpy(y_np).long()         # (N,)
        pose = {"ego_x": fm.ego_x, "ego_y": fm.ego_y, "ego_z": fm.ego_z, "ego_yaw": fm.ego_yaw}
        return Sample(x=x, y=y, pose=pose)


def build_frames_from_dir(data_dir: str, data_cfg: DataConfig) -> List[FrameMeta]:
    """
    Construit/charge un index par .h5 puis concatène toutes les frames.
    """
    data_path = Path(data_dir)
    cache_dir = Path(data_cfg.index_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames: List[FrameMeta] = []
    for h5_path in sorted(data_path.glob("scene_*.h5")):
        cache_path = cache_dir / f"{h5_path.stem}_index.pkl"
        if cache_path.exists():
            idx = H5FrameIndex.load(str(cache_path))
        else:
            idx = H5FrameIndex(str(h5_path), dataset_name=data_cfg.dataset_name).build()
            idx.save(str(cache_path))
        frames.extend(idx.frames)
    return frames
