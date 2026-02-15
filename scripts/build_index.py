from pathlib import Path
from src.airbus_lidar.config import DataConfig
from src.airbus_lidar.data.dataset import build_frames_from_dir

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]  # racine projet
    data_dir = root / "airbus_hackathon_trainingdata"
    print("Data dir:", data_dir)
    print("H5 found:", list(data_dir.glob("scene_*.h5")))

    cfg = DataConfig(train_data_dir=str(data_dir))
    frames = build_frames_from_dir(cfg.train_data_dir, cfg)
    print(f"Indexed frames: {len(frames)}")
    if frames:
        print("Example:", frames[0])
