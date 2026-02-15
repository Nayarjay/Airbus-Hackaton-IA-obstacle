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
    infer_cfg.checkpoint_path = "checkpoints/pointnet_seg_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_cfg.device = device.type

    in_ch = 4 if data_cfg.use_intensity else 3
    model = PointNetSeg(in_channels=in_ch, num_classes=NUM_CLASSES)
    model = load_model(infer_cfg.checkpoint_path, model, device=device)

    eval_dir = Path("../airbus_hackathon_evalset")
    out_dir = Path("predictions_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "eval_sceneA_100.h5", "eval_sceneA_75.h5", "eval_sceneA_50.h5", "eval_sceneA_25.h5",
        "eval_sceneB_100.h5", "eval_sceneB_75.h5", "eval_sceneB_50.h5", "eval_sceneB_25.h5",
    ]

    for name in files:
        h5_path = eval_dir / name
        if not h5_path.exists():
            print(f"SKIP missing: {h5_path}")
            continue

        rows = infer_file_to_rows(str(h5_path), model, data_cfg, infer_cfg)

        # nom de sortie explicite
        out_csv = out_dir / (h5_path.stem + "_pred.csv")
        write_predictions_csv(str(out_csv), rows)
        print(f"Wrote {len(rows)} rows -> {out_csv}")