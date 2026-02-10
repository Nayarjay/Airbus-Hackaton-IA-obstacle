# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Airbus AI Hackathon 2026 project for **LiDAR obstacle detection**. The goal is to detect and classify 3D obstacles (antennas, cables, electric poles, wind turbines) from simulated LiDAR point clouds for helicopter safety.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Visualize a LiDAR frame (list available poses first)
python visualize.py --file <path_to_file.h5>

# Visualize specific pose/frame
python visualize.py --file <path_to_file.h5> --pose-index <N>

# Train PointNet segmentation model
python train.py --data_dir ./data --epochs 100 --batch_size 8 --n_points 32768

# Run inference and generate CSV predictions
python inference.py --model checkpoints/best.pth --input <input.h5> --output predictions.csv
```

## Architecture

### PointNet Segmentation Pipeline
- **pointnet_model.py**: PointNet segmentation architecture (~3.5M parameters)
  - `TNet`: Transformation network for spatial/feature alignment
  - `PointNetEncoder`: Shared MLPs (64→128→1024) + global max pooling
  - `PointNetSegmentation`: Full segmentation model (5 classes output)
- **dataset.py**: PyTorch Dataset for LiDAR data with augmentation
- **train.py**: Training script with class-weighted loss and IoU metrics
- **inference.py**: Inference with DBSCAN clustering and 3D bounding box computation

### Data Utilities
- **lidar_utils.py**: Core library for data loading and coordinate conversion
  - `load_h5_data(file_path)`: Loads HDF5 LiDAR data into pandas DataFrame
  - `get_unique_poses(df)`: Extracts unique frames by ego pose (x, y, z, yaw)
  - `filter_by_pose(df, pose_row)`: Filters points for a specific frame
  - `spherical_to_local_cartesian(df)`: Converts raw units (cm, 1/100 deg) to local Cartesian meters

- **visualize.py**: Open3D-based point cloud visualization tool

## Data Format

HDF5 files contain LiDAR points with fields:
- `distance_cm`, `azimuth_raw`, `elevation_raw`: Spherical coordinates (cm, 1/100 deg)
- `reflectivity`: Laser return intensity (0-255)
- `r, g, b`: Ground truth class labels encoded as RGB
- `ego_x, ego_y, ego_z, ego_yaw`: Vehicle pose (frame identifier)

### Class Labels (RGB encoding)

| Class | Label         | R   | G   | B   |
|-------|---------------|-----|-----|-----|
| 0     | Antenna       | 38  | 23  | 180 |
| 1     | Cable         | 177 | 132 | 47  |
| 2     | Electric pole | 129 | 81  | 97  |
| 3     | Wind turbine  | 66  | 132 | 9   |

## Key Constraints

- Ground truth 3D bounding boxes are NOT provided; must be reconstructed from point-wise labels
- Valid points have `distance_cm > 0`; zero distance indicates invalid beams
- Frames are identified by the unique quadruplet `(ego_x, ego_y, ego_z, ego_yaw)`
- Training data: 10 scenes, 100 frames total, up to 575k points per frame
