import os
from pathlib import Path
import torch

from src.airbus_lidar.config import DataConfig, InferConfig
from src.airbus_lidar.constants import NUM_CLASSES
from src.airbus_lidar.models.pointnet_seg import PointNetSeg
from src.airbus_lidar.infer.inference import load_model, infer_file_to_rows
from src.airbus_lidar.infer.export_csv import write_predictions_csv


if __name__ == "__main__":
    data_cfg = DataConfig()
    infer_cfg = InferConfig()

    # Chemin absolu (évite les soucis de working directory)
    root = Path(__file__).resolve().parents[1]  # racine projet
    data_dir = root / "airbus_hackathon_trainingdata"
    data_cfg.train_data_dir = str(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_cfg.device = device.type

    in_ch = 4 if data_cfg.use_intensity else 3
    model = PointNetSeg(in_channels=in_ch, num_classes=NUM_CLASSES)
    model = load_model(infer_cfg.checkpoint_path, model, device=device)

    # exemple: inférer sur un fichier (train ou eval)
    h5_path = str(data_dir / "scene_1.h5")
    rows = infer_file_to_rows(h5_path, model, data_cfg, infer_cfg)

    out_csv = str(root / "pred_scene_1.csv")
    write_predictions_csv(out_csv, rows)
    print(f"Wrote {len(rows)} bbox rows to {out_csv}")
