# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Airbus AI Hackathon 2026 project for **LiDAR obstacle detection**. The goal is to detect and classify 3D obstacles (antennas, cables, electric poles, wind turbines) from simulated LiDAR point clouds for helicopter safety.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Visualize a LiDAR frame (list available poses first)
python visualize.py --file data/scene_1.h5

# Visualize specific pose/frame
python visualize.py --file data/scene_1.h5 --pose-index 0

# Training: Use LiDAR_Training_Colab.ipynb on Google Colab with GPU
```

## Architecture

### Data Pipeline
1. **Load**: `load_h5_data()` → raw HDF5 to DataFrame (spherical coords in cm, 1/100 deg)
2. **Filter**: `get_unique_poses()` + `filter_by_pose()` → extract single frame by ego pose
3. **Convert**: `spherical_to_local_cartesian()` → local Cartesian (meters)
4. **Labels**: RGB values map to class IDs (4 classes + background)

### Model (PointNet Segmentation)
- **Input**: Point cloud tensor `[batch, n_points, 4]` (x, y, z, reflectivity)
- **Architecture**: TNet transforms → shared MLPs → global pooling → per-point segmentation
- **Output**: Per-point class predictions `[batch, n_points, 5]`
- **Parameters**: ~1.8M trainable (target: low param count for efficiency scoring)

### Training Configuration (Colab notebook)
- 32k points sampled per frame, batch size 8
- Weighted cross-entropy loss (handles class imbalance)
- Cosine annealing LR scheduler, 100 epochs
- Data augmentation: random rotation, jitter, scaling

## Existing Code

- **lidar_utils.py**: Core data loading and coordinate conversion
- **visualize.py**: Open3D point cloud viewer
- **LiDAR_Training_Colab.ipynb**: Full training pipeline with PointNet model

## Data Format

HDF5 files (`data/scene_*.h5`) with fields:
- `distance_cm`, `azimuth_raw`, `elevation_raw`: Spherical coords (cm, 1/100 deg)
- `reflectivity`: Laser intensity (0-255)
- `r, g, b`: Ground truth class labels
- `ego_x, ego_y, ego_z, ego_yaw`: Vehicle pose (frame ID)

### Class Labels (RGB → Class ID)

| ID | Label         | R   | G   | B   |
|----|---------------|-----|-----|-----|
| 0  | Antenna       | 38  | 23  | 180 |
| 1  | Cable         | 177 | 132 | 47  |
| 2  | Electric pole | 129 | 81  | 97  |
| 3  | Wind turbine  | 66  | 132 | 9   |
| 4  | Background    | any other RGB       |

## Key Constraints

- Ground truth 3D bounding boxes NOT provided; reconstruct from point-wise labels
- Valid points: `distance_cm > 0`
- Evaluation data will have `r=g=b=128` (no labels)
- 10 training scenes, 100 frames, up to 575k points/frame

## Deliverables

1. **Model**: ONNX or PyTorch checkpoint
2. **Training code**: With `requirements.txt`
3. **Inference code**: Output CSV with columns:
   - `ego_x`, `ego_y`, `ego_z`, `ego_yaw` (frame ID)
   - `bbox_center_x/y/z` (meters)
   - `bbox_width/length/height` (meters, before yaw rotation)
   - `bbox_yaw` (Z-axis rotation)
   - `Class ID`, `Class Label`
4. **8 prediction CSVs**: For 2 scenes × 4 density levels (100%, 75%, 50%, 25%)
5. **Evaluation metrics**: mAP @ IoU=0.5, robustness to density reduction
