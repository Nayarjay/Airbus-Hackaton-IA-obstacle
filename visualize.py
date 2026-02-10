import argparse
import h5py
import numpy as np
import open3d as o3d
import os
import torch
import lidar_utils
import inference_utils
from model_pointnet import PointNetSeg

# Color Map (same as test_interactive)
CLASS_COLORS = np.array([
    [0.15, 0.09, 0.70], # Antenna
    [0.70, 0.52, 0.18], # Cable
    [0.50, 0.32, 0.38], # Pole
    [0.26, 0.52, 0.04], # Turbine
    [0.50, 0.50, 0.50], # Background
])

def find_scenes(file_path):
    """Detects scene boundaries in an H5 file."""
    with h5py.File(file_path, 'r') as f:
        ds = f['lidar_points']
        N = ds.shape[0]
        ego_x = ds['ego_x'][:]
        diff = np.abs(np.diff(ego_x))
        change_indices = np.where(diff > 100)[0] + 1
        indices = np.concatenate(([0], change_indices, [N]))
        
        scenes = []
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] > 100:
                scenes.append((indices[i], indices[i+1]))
        return scenes

def visualize_frame(file_path, pose_index, model_path=None):
    scenes = find_scenes(file_path)
    if pose_index >= len(scenes):
        print(f"Error: Pose index {pose_index} out of range (Found {len(scenes)} scenes in file).")
        return

    start, end = scenes[pose_index]
    print(f"Visualizing scene {pose_index} (points {start} to {end}) from {os.path.basename(file_path)}")

    with h5py.File(file_path, 'r') as f:
        ds = f['lidar_points']
        data = ds[start:end]
        
    # Coordinate Conversion
    xyz = lidar_utils.spherical_to_local_cartesian({
        "distance_cm": data['distance_cm'],
        "azimuth_raw": data['azimuth_raw'],
        "elevation_raw": data['elevation_raw']
    })
    
    # GT Labels from RGB
    r, g, b = data['r'], data['g'], data['b']
    labels = np.full(len(data), 4, dtype=np.int64)
    labels[(r == 38) & (g == 23) & (b == 180)] = 0
    labels[(r == 177) & (g == 132) & (b == 47)] = 1
    labels[(r == 129) & (g == 81) & (b == 97)] = 2
    labels[(r == 66) & (g == 132) & (b == 9)] = 3
    
    # Prediction (Optional)
    pred_labels = labels # Default to GT if no model
    if model_path and os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PointNetSeg(k=5).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Sub-sample for model (1024 points)
        N = len(xyz)
        indices = np.random.choice(N, 1024, replace=(N < 1024))
        xyz_sub = xyz[indices]
        
        # Norm
        xyz_norm = xyz_sub - np.mean(xyz_sub, axis=0)
        max_d = np.max(np.linalg.norm(xyz_norm, axis=1))
        if max_d > 0: xyz_norm /= max_d
        
        inp = torch.from_numpy(xyz_norm).float().transpose(1, 0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred, _ = model(inp)
            pred_labels_sub = pred.data.max(2)[1].cpu().numpy()[0]
            
        print("Model prediction complete for subsample.")
        # For full vis, we just use GT colors for points but we can show BBoxes
        # Better: just use full points and show boxes
    
    # Visualization setup
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(CLASS_COLORS[labels])
    
    # Bounding Boxes
    boxes_data = inference_utils.get_boxes_from_segmentation(xyz, labels)
    bbox_geos = []
    for box in boxes_data:
        color = CLASS_COLORS[box["class_id"]]
        bbox_geos.append(inference_utils.create_o3d_box_visual(box["center"], box["dims"], color))
    
    print(f"Showing {len(bbox_geos)} objects.")
    o3d.visualization.draw_geometries([pcd] + bbox_geos, window_name=f"Pose {pose_index} - {os.path.basename(file_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to H5 file")
    parser.add_argument("--pose-index", type=int, default=0, help="Index of the pose/scene to visualize")
    parser.add_argument("--model", type=str, default="models/pointnet_segmentation.pth", help="Path to model for prediction")
    args = parser.parse_args()
    
    visualize_frame(args.file, args.pose_index, args.model)