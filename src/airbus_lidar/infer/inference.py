from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Sequence

import numpy as np
import torch

from src.airbus_lidar.config import DataConfig, InferConfig
from src.airbus_lidar.constants import BACKGROUND_ID
from src.airbus_lidar.io.h5_index import H5FrameIndex, FrameMeta
from src.airbus_lidar.io.h5_reader import read_frame_fields
from src.airbus_lidar.geometry.coords import spherical_to_local_cartesian_np
from src.airbus_lidar.geometry.bbox import cluster_and_build_bboxes
from src.airbus_lidar.data.dataset import rgb_to_class_id


def load_model(checkpoint_path: str, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def infer_file_to_rows(
    h5_path: str,
    model: torch.nn.Module,
    data_cfg: DataConfig,
    infer_cfg: InferConfig,
) -> List[Dict]:
    # index frames
    idx = H5FrameIndex(h5_path, dataset_name=data_cfg.dataset_name).build()

    fields = ["distance_cm", "azimuth_raw", "elevation_raw", "reflectivity",
              "ego_x", "ego_y", "ego_z", "ego_yaw"]

    all_rows: List[Dict] = []
    device = torch.device(infer_cfg.device)

    for fm in idx.frames:
        d = read_frame_fields(h5_path, fm.start, fm.end, data_cfg.dataset_name, fields)

        valid = d["distance_cm"] > 0
        if not np.any(valid):
            continue

        dist = d["distance_cm"][valid]
        az = d["azimuth_raw"][valid]
        el = d["elevation_raw"][valid]
        xyz = spherical_to_local_cartesian_np(dist, az, el)  # (N,3)

        if data_cfg.use_intensity:
            inten = d["reflectivity"][valid].astype(np.float32) / 255.0
            feats = np.concatenate([xyz, inten[:, None]], axis=1)  # (N,4)
        else:
            feats = xyz  # (N,3)

        # tensors
        x_full = torch.from_numpy(feats.T).unsqueeze(0).float().to(device)  # (1,C,N)

        # subset pour global feature
        N = feats.shape[0]
        Ng = min(data_cfg.num_points_global, N)
        idxg = np.random.choice(N, size=Ng, replace=(N < Ng))
        x_global = x_full[:, :, idxg]

        # logits full cloud
        logits = model.predict_full_cloud(
            x_full=x_full,
            x_global=x_global,
            chunk_size=infer_cfg.batch_points_chunk,
        )  # (1,K,N)

        probs = torch.softmax(logits, dim=1)
        conf, pred_t = probs.max(dim=1)  # (1,N)
        pred = pred_t.squeeze(0).detach().cpu().numpy().astype(np.int64)
        conf = conf.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # garder uniquement classes 0..3
        mask_obs = (pred != BACKGROUND_ID) & (conf >= 0.75)
        if not np.any(mask_obs):
            continue

        xyz_obs = xyz[mask_obs]
        pred_obs = pred[mask_obs]

        bboxes = cluster_and_build_bboxes(xyz_obs, pred_obs, infer_cfg.cluster)

        pose = {"ego_x": fm.ego_x, "ego_y": fm.ego_y, "ego_z": fm.ego_z, "ego_yaw": fm.ego_yaw}

        # rows CSV
        from src.airbus_lidar.infer.export_csv import bboxes_to_rows
        all_rows.extend(bboxes_to_rows(pose, bboxes))

    return all_rows
