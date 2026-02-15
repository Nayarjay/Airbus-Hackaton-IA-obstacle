from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DataConfig:
    dataset_name: str = "lidar_points"
    train_data_dir: str = "airbus_hackathon_trainingdata"
    index_cache_dir: str = ".cache_indexes"

    num_points_train: int = 8192
    num_points_global: int = 8192   # pour global feature en inference "full cloud"
    use_intensity: bool = True

    # train/val split sur frames (ex: 0.9)
    train_split: float = 0.9
    seed: int = 42


@dataclass
class TrainConfig:
    epochs: int = 70
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    amp: bool = True  # mixed precision (3070Ti ok)
    log_every: int = 20
    save_dir: str = "checkpoints"
    run_name: str = "pointnet_seg"


@dataclass
class ClusterConfig:
    eps_by_class: Dict[int, float] = None
    min_samples_by_class: Dict[int, int] = None
    min_cluster_points: int = 10  # ↓ pour éviter de perdre des objets fins (câbles)

    # sécurité mémoire DBSCAN
    max_points_by_class: Dict[int, int] = None

    # voxel downsample (m)
    voxel_size_by_class: Dict[int, float] = None

    # NEW: marge multiplicative sur la bbox (par classe)
    inflate_by_class: Dict[int, float] = None

    # NEW: dimensions minimales (width, length, height) en mètres, par classe
    min_extent_by_class: Dict[int, tuple] = None

    def __post_init__(self):
        # DBSCAN plus permissif pour récupérer blades/haut pylône + câbles
        if self.eps_by_class is None:
            self.eps_by_class = {
                0: 1.2,  # Antenna
                1: 1.3,  # Cable
                2: 1.8,  # Electric pole
                3: 4.0,  # Wind turbine
            }
        if self.min_samples_by_class is None:
            self.min_samples_by_class = {
                0: 8,
                1: 3,   # ↓ important pour câble
                2: 5,
                3: 5,
            }

        # on évite l'allocation énorme DBSCAN
        if self.max_points_by_class is None:
            self.max_points_by_class = {
                0: 60000,
                1: 150000,  # câble peut être long
                2: 80000,
                3: 120000,
            }

        # voxel plus fin pour garder la géométrie des câbles / objets
        if self.voxel_size_by_class is None:
            self.voxel_size_by_class = {
                0: 0.10,
                1: 0.03,  # câble fin
                2: 0.08,
                3: 0.12,
            }

        # marge bbox (englobe mieux)
        if self.inflate_by_class is None:
            self.inflate_by_class = {
                0: 1.15,
                1: 1.10,
                2: 1.20,
                3: 1.25,
            }

        # épaisseurs minimales pour visibilité + robustesse IoU
        if self.min_extent_by_class is None:
            self.min_extent_by_class = {
                1: (0.25, 1.0, 0.25),  # câble: min width/height 25cm, min length 1m
            }


@dataclass
class InferConfig:
    checkpoint_path: str = "checkpoints/pointnet_seg_best.pt"
    device: str = "cuda"
    batch_points_chunk: int = 65536
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
