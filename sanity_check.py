# sanity_check.py
import glob
import torch
import lidar_utils

def main():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    files = glob.glob("data/train/*.h5") + glob.glob("data/eval/*.h5")
    if not files:
        print("No h5 files found in data/train or data/eval")
        return

    df = lidar_utils.load_h5_data(files[0])
    print("Loaded:", files[0])
    print("Columns:", list(df.columns)[:30], "...")
    poses = lidar_utils.get_unique_poses(df)
    print("Poses:", len(poses))
    print(poses.head(3))

if __name__ == "__main__":
    main()
