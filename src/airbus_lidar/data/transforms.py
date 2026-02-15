from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainAugment:
    rotate_z: bool = True
    jitter_std: float = 0.01
    scale_min: float = 0.95
    scale_max: float = 1.05
    dropout_ratio: float = 0.0  # si tu veux du point dropout au train

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        out = xyz.astype(np.float32)

        if self.rotate_z:
            yaw = np.random.uniform(-np.pi, np.pi)
            c = float(np.cos(yaw))
            s = float(np.sin(yaw))
            R = np.array([[c, -s, 0.0],
                          [s,  c, 0.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
            out = (out @ R.T).astype(np.float32)

        scale = np.random.uniform(self.scale_min, self.scale_max)
        out *= np.float32(scale)

        if self.jitter_std > 0:
            out += np.random.normal(0.0, self.jitter_std, size=out.shape).astype(np.float32)

        return out


def sample_or_pad_indices(n: int, k: int) -> np.ndarray:
    if n >= k:
        return np.random.choice(n, size=k, replace=False)
    # pad: on répète des indices
    extra = np.random.choice(n, size=(k - n), replace=True)
    base = np.arange(n)
    return np.concatenate([base, extra], axis=0)
