import torch
import numpy as np
from model_pointnet import PointNetCls
import lidar_utils

class ObstacleAvoider:
    def __init__(self, model_path='models/pointnet_obstacle_avoidance.pth', device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = PointNetCls(k=2).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using uninitialized model.")
        self.model.eval()

    def preprocess(self, xyz, n_points=1024):
        """
        Preprocesses a numpy array of points (N, 3) for the model.
        """
        # Resampling
        if len(xyz) >= n_points:
            choice = np.random.choice(len(xyz), n_points, replace=False)
        else:
            choice = np.random.choice(len(xyz), n_points, replace=True)
        
        xyz = xyz[choice, :]
        
        # Normalize
        xyz = xyz - np.mean(xyz, axis=0)
        max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
            
        # Transpose and add batch dimension
        xyz = xyz.T.astype(np.float32)
        xyz_tensor = torch.from_numpy(xyz).unsqueeze(0).to(self.device)
        return xyz_tensor

    def predict(self, xyz):
        """
        Returns 0 (SAFE) or 1 (DANGER)
        """
        xyz_tensor = self.preprocess(xyz)
        with torch.no_grad():
            pred, _ = self.model(xyz_tensor)
            pred_choice = pred.data.max(1)[1].item()
        
        return pred_choice

if __name__ == "__main__":
    # Example usage with a scene pose
    avoider = ObstacleAvoider()
    
    # Load some real data for testing
    DATA_FILE = r"airbus_hackathon_trainingdata\scene_1.h5"
    df = lidar_utils.load_h5_data(DATA_FILE)
    pose_counts = lidar_utils.get_unique_poses(df)
    
    if pose_counts is not None:
        selected_pose = pose_counts.iloc[0]
        pts_df = lidar_utils.filter_by_pose(df, selected_pose)
        xyz = lidar_utils.spherical_to_local_cartesian(pts_df)
        
        decision = avoider.predict(xyz)
        labels = ["SAFE", "DANGER"]
        print(f"Decision for Pose #0: {labels[decision]}")
