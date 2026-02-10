# test_interactive.py (Segmentation Version)
import torch
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from dataset import LidarH5Dataset
from model_pointnet import PointNetSeg
import lidar_utils
import sys
import inference_utils # New import

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
        self.boxes = []
        self.view_mode = "prediction"
        self.auto_mode = False # New state for Hunt Mode
        self.current_data = None
        self.current_pred = None

def check_discrepancy(gt_labels, pred_labels):
    """
    Detective logic: Returns True if there's a big gap between Reality and Prediction.
    """
    for cls_id in range(4): # For each obstacle class
        gt_count = np.sum(gt_labels == cls_id)
        pred_count = np.sum(pred_labels == cls_id)
        
        # Scenario 1: Object exists but IA missed most of it
        if gt_count > 20 and pred_count < gt_count * 0.3:
            print(f"!!! DISCREPANCY: Missed {CLASS_NAMES[cls_id]} (GT:{gt_count}, Pred:{pred_count})")
            return True
            
        # Scenario 2: IA hallucinated an object
        if gt_count < 5 and pred_count > 30:
            print(f"!!! DISCREPANCY: Hallucinated {CLASS_NAMES[cls_id]} (GT:{gt_count}, Pred:{pred_count})")
            return True
            
    return False

def update_visualization(vis, state, points_tensor, label_tensor, pred_choice):
    xyz = points_tensor.numpy().T
    label_numpy = label_tensor.numpy()
    state.pcd.points = o3d.utility.Vector3dVector(xyz)
    
    for box in state.boxes:
        vis.remove_geometry(box, reset_bounding_box=False)
    state.boxes = []
    
    if state.view_mode == "prediction":
        state.pcd.colors = o3d.utility.Vector3dVector(CLASS_COLORS[pred_choice])
        mode_text = "[PREDICTION]"
        boxes_data = inference_utils.get_boxes_from_segmentation(xyz, pred_choice)
    else:
        state.pcd.colors = o3d.utility.Vector3dVector(CLASS_COLORS[label_numpy])
        mode_text = "[GROUND TRUTH]"
        boxes_data = inference_utils.get_boxes_from_segmentation(xyz, label_numpy)
        
    for box_info in boxes_data:
        box_geo = inference_utils.create_o3d_box_visual(box_info["center"], box_info["dims"], CLASS_COLORS[box_info["class_id"]])
        vis.add_geometry(box_geo, reset_bounding_box=False)
        state.boxes.append(box_geo)

    vis.update_geometry(state.pcd)
    
    gt_stats = {CLASS_NAMES[k]: v for k, v in dict(zip(*np.unique(label_numpy, return_counts=True))).items() if k < 4}
    pred_stats = {CLASS_NAMES[k]: v for k, v in dict(zip(*np.unique(pred_choice, return_counts=True))).items() if k < 4}
    
    print(f"\n--- Sample [{state.index}] {mode_text} ---")
    if gt_stats: print(f"  Reality: {gt_stats}")
    if pred_stats: print(f"  AI Saw : {pred_stats}")

def next_sample(vis, state, force_interesting=True):
    sample_found = False
    while not sample_found:
        if state.index >= len(state.dataset):
            state.index = 0
            print("\n--- Loop ---")
            
        points_tensor, label_tensor = state.dataset[state.index]
        if not force_interesting or (label_tensor < 4).any().item():
            sample_found = True
        else:
            state.index += 1
            if state.index % 100 == 0: sample_found = True
                
    state.current_data = (points_tensor, label_tensor)
    points_input = points_tensor.unsqueeze(0).to(state.device)
    with torch.no_grad():
        pred, _ = state.model(points_input)
        state.current_pred = pred.data.max(2)[1].cpu().numpy()[0]
    
    update_visualization(vis, state, points_tensor, label_tensor, state.current_pred)
    state.index += 1
    return True

def main():
    DATA_PATH = r"airbus_hackathon_trainingdata"
    MODEL_PATH = "models/pointnet_segmentation.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PointNetSeg(k=5).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("Model loaded.")
    
    model.eval()
    dataset = LidarH5Dataset(DATA_PATH, n_points=1024, use_cache=True)
    state = ViewerState(dataset, model, device)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="AI Hunter Viewer", width=1280, height=720)
    
    # Init geometry
    xyz_init, _ = dataset[0]
    state.pcd.points = o3d.utility.Vector3dVector(xyz_init.numpy().T)
    vis.add_geometry(state.pcd)

    def callback_next(vis):
        state.auto_mode = False # Kill auto if manual skip
        next_sample(vis, state)
        return True

    def callback_toggle_gt(vis):
        state.view_mode = "ground_truth" if state.view_mode == "prediction" else "prediction"
        if state.current_data:
            update_visualization(vis, state, state.current_data[0], state.current_data[1], state.current_pred)
        return True

    def callback_hunt(vis):
        state.auto_mode = not state.auto_mode
        print(f"\n>>> HUNT MODE: {'ENABLED' if state.auto_mode else 'DISABLED'}")
        return True

    def animation_callback(vis):
        if state.auto_mode:
            # We are in hunt mode. Next sample...
            next_sample(vis, state, force_interesting=False)
            
            # Check if this sample is a "failure"
            is_fail = check_discrepancy(state.current_data[1].numpy(), state.current_pred)
            if is_fail:
                state.auto_mode = False
                print(">>> STOPPED! Significant discrepancy found.")
                # Flash the screen/window? Just stop is enough.
        return False # Return False to keep UI alive

    vis.register_key_callback(ord('N'), callback_next)
    vis.register_key_callback(ord('G'), callback_toggle_gt)
    vis.register_key_callback(ord('A'), callback_hunt)
    vis.register_animation_callback(animation_callback)
    
    print("\nControls:")
    print("  [N] : Next Sample (Manual)")
    print("  [G] : Toggle GT/Prediction")
    print("  [A] : Toggle AUTO-HUNT (Stop on discrepancy)")
    
    next_sample(vis, state)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    import os
    main()

if __name__ == "__main__":
    import os
    main()
