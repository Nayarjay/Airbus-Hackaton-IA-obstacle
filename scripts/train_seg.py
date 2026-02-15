import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.airbus_lidar.config import DataConfig, TrainConfig
from src.airbus_lidar.data.dataset import build_frames_from_dir, LidarFrameDataset
from src.airbus_lidar.data.transforms import TrainAugment
from src.airbus_lidar.data import collate_samples
from src.airbus_lidar.constants import NUM_CLASSES
from src.airbus_lidar.models.pointnet_seg import PointNetSeg
from src.airbus_lidar.train.trainer import Trainer


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    data_cfg = DataConfig()
    train_cfg = TrainConfig()
    seed_all(data_cfg.seed)

    # Chemin absolu (évite les problèmes de working directory)
    root = Path(__file__).resolve().parents[1]  # racine projet
    data_dir = root / "airbus_hackathon_trainingdata"
    data_cfg.train_data_dir = str(data_dir)

    frames = build_frames_from_dir(data_cfg.train_data_dir, data_cfg)
    print("Data dir:", data_cfg.train_data_dir)
    print("Total frames:", len(frames))
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {data_cfg.train_data_dir}")

    # split simple
    rng = np.random.default_rng(data_cfg.seed)
    perm = rng.permutation(len(frames))
    split = int(len(frames) * data_cfg.train_split)
    train_frames = [frames[i] for i in perm[:split]]
    val_frames = [frames[i] for i in perm[split:]]

    print("Train frames:", len(train_frames), "Val frames:", len(val_frames))

    train_ds = LidarFrameDataset(train_frames, data_cfg, train=True, augment=TrainAugment())
    val_ds = LidarFrameDataset(val_frames, data_cfg, train=False, augment=None)

    print("Train ds:", len(train_ds), "Val ds:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_samples,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_samples,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    in_ch = 4 if data_cfg.use_intensity else 3
    model = PointNetSeg(in_channels=in_ch, num_classes=NUM_CLASSES).to(device)

    ckpt = torch.load("checkpoints/pointnet_seg_best.pt", map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay
    )

    optimizer.load_state_dict(ckpt["optimizer"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        amp=train_cfg.amp and device.type == "cuda",
        save_dir=train_cfg.save_dir,
        run_name=train_cfg.run_name,
    )
    trainer.fit(train_cfg.epochs)
