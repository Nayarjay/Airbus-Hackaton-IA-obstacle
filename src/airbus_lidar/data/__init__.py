from __future__ import annotations
from typing import List, Dict, Any
import torch
from src.airbus_lidar.data.dataset import Sample


def collate_samples(batch: List[Sample]) -> Dict[str, Any]:
    x = torch.stack([b.x for b in batch], dim=0)  # (B,C,N)
    y = torch.stack([b.y for b in batch], dim=0)  # (B,N)
    pose = [b.pose for b in batch]
    return {"x": x, "y": y, "pose": pose}
