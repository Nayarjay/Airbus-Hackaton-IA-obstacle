from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import h5py
import numpy as np


def read_frame_fields(
    file_path: str,
    start: int,
    end: int,
    dataset_name: str,
    fields: Sequence[str],
) -> Dict[str, np.ndarray]:
    start = int(start)
    end = int(end)
    with h5py.File(file_path, "r") as f:
        dset = f[dataset_name]

        # IMPORTANT: sélectionner les champs AVANT de slicer
        rec = dset.fields(list(fields))[start:end]  # structured array (end-start,)

    return {k: rec[k] for k in fields}



def read_frame_struct(
    file_path: str,
    start: int,
    end: int,
    dataset_name: str,
) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        dset = f[dataset_name]
        return dset[start:end]
