# test_interactive.py (Segmentation Version)
import torch
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from dataset import LidarH5Dataset
from model_pointnet import PointNetSeg
import lidar_utils
import sys

# Color Map for 5 Classes
# 0: Antenna (Purple), 1: Cable (Orange), 2: Pole (Brown), 3: Turbine (Green), 4: Background (Gray/Blue)
CLASS_COLORS = np.array([
    [0.15, 0.09, 0.70], # Antenna
    [0.70, 0.52, 0.18], # Cable (Orange-ish)
    [0.50, 0.32, 0.38], # Pole
    [0.26, 0.52, 0.04], # Turbine
    [0.50, 0.50, 0.50], # Background (Gray)
])

CLASS_NAMES = ["Antenna", "Cable", "Pole", "Turbine", "Background"]

class ViewerState:
    def __init__(self, dataset, model, device):
        self.index = 0
        self.dataset = dataset
        self.model = model
        self.device = device
        self.pcd = o3d.geometry.PointCloud()
        self.view_mode = "prediction" # or "ground_truth"

def next_sample(vis, state):
    sample_found = False
    
    # Fast-forward loop to find interesting samples (containing obstacles)
    while not sample_found:
        if state.index >= len(state.dataset):
            state.index = 0
            print("--- Loop ---")
            
        print(f"Loading sample {state.index}...", end='\r')
        
        # Get data
        points_tensor, label_tensor = state.dataset[state.index]
        
        # Check if sample has any obstacle (class < 4)
        has_obstacle = (label_tensor < 4).any().item()
        
        if not has_obstacle:
            state.index += 1
            if state.index % 50 == 0:
                # Force show every 50th background sample just to confirm it works
                sample_found = True
            continue
        else:
            sample_found = True

    # Predict
    points_input = points_tensor.unsqueeze(0).to(state.device)
    with torch.no_grad():
        pred, _ = state.model(points_input) # (1, N, 5)
        pred_choice = pred.data.max(2)[1].cpu().numpy()[0] # (N,)
    
    # Update Geometry
    xyz = points_tensor.numpy().T
    label_numpy = label_tensor.numpy()
    
    state.pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Colorize
    # Use Prediction or Ground Truth based on mode
    # Let's show: Left = GT, Right = Pred? OR just toggle?
    # For now, let's mix: 
    # Points with obstacles are colored by class.
    # Background points are gray.
    
    colors = CLASS_COLORS[pred_choice]
    state.pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if len(state.pcd.points) > 0:
        vis.update_geometry(state.pcd)
    
    vis.poll_events()
    vis.update_renderer()
    
    # Console Stats
    unique, counts = np.unique(label_numpy, return_counts=True)
    gt_stats = dict(zip(unique, counts))
    
    unique_p, counts_p = np.unique(pred_choice, return_counts=True)
    pred_stats = dict(zip(unique_p, counts_p))
    
    print(f"\n--- Sample [{state.index}] ---")
    print("Ground Truth Objects:")
    for cls_id, count in gt_stats.items():
        if cls_id < 4:
            print(f"  - {CLASS_NAMES[cls_id]}: {count} points")
            
    print("Prediction:")
    for cls_id, count in pred_stats.items():
        if cls_id < 4:
            print(f"  - {CLASS_NAMES[cls_id]}: {count} points")
    print("-" * 20)
    
    state.index += 1
    return False

def main():
    DATA_PATH = r"airbus_hackathon_trainingdata"
    MODEL_PATH = "models/pointnet_segmentation.pth"
    N_POINTS = 1024
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = PointNetSeg(k=5, feature_transform=True).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Allow running without model for checking dataset
        # return 
        
    model.eval()

    # Load Dataset
    print("Loading dataset...")
    dataset = LidarH5Dataset(DATA_PATH, n_points=N_POINTS, use_cache=True)
    print(f"Dataset has {len(dataset)} samples.")

    # Visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="AI Segmentation Viewer (Press 'N')", width=1280, height=720)

    state = ViewerState(dataset, model, device)
    
    # Dummy init
    state.pcd.points = o3d.utility.Vector3dVector(np.random.rand(N_POINTS, 3))
    vis.add_geometry(state.pcd)

    def callback_wrapper(vis):
        return next_sample(vis, state)

    vis.register_key_callback(ord('N'), callback_wrapper)
    
    print("\nControls:")
    print("  [N] : Next Interesting Sample")
    
    # Try to load first sample
    try:
        next_sample(vis, state)
    except Exception as e:
        print(f"Initial load failed (maybe model shape mismatch if not trained yet): {e}")

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
