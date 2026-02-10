import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lidar_utils
from tqdm import tqdm

class LidarH5Dataset(Dataset):
    def __init__(self, data_dir, n_points=1024, critical_dist=2.0, scan_angle_front=60, use_cache=True):
        """
        Args:
            data_dir (str): Directory containing .h5 files.
            n_points (int): Number of points to sample per pose.
            critical_dist (float): Distance in meters to trigger 'DANGER' label.
            scan_angle_front (float): Angle in degrees (left/right of front) to check for obstacles.
            use_cache (bool): If True, loads all relevant points into RAM at startup.
        """
        self.data_dir = data_dir
        self.n_points = n_points
        self.critical_dist = critical_dist
        self.scan_angle_front = scan_angle_front
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        
        self.cached_samples = [] # List of (xyz_tensor, label)
        self._load_data(use_cache)

    def _load_data(self, use_cache):
        print(f"Indexing {len(self.files)} H5 files efficiently...")
        self.samples_info = [] # List of (file_path, start_idx, end_idx)

        # Define Color Mapping (R, G, B) -> Class ID
        # Class 0: Antenna (38, 23, 180)
        # Class 1: Cable (177, 132, 47)
        # Class 2: Electric pole (129, 81, 97)
        # Class 3: Wind turbine (66, 132, 9)
        # Everything else: Background (4)

        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    ds = f['lidar_points']
                    N = ds.shape[0]
                    
                    # Read ego columns to find scene boundaries
                    # Optim: Read ego_x (float) to detect jumps
                    ego_x = ds['ego_x'][:]
                    
                    # Find indices where ego_x changes significantly (> 1cm jump usually means next frame or just diff frame)
                    # Actually, frames are sequential. Let's look for step changes.
                    diff = np.abs(np.diff(ego_x))
                    # A Jump in X usually indicates a new "teleport" or frame start
                    # Threshold: 100 cm (1m) jump
                    change_indices = np.where(diff > 100)[0] + 1
                    
                    indices = np.concatenate(([0], change_indices, [N]))
                    
                    for i in range(len(indices) - 1):
                        start = indices[i]
                        end = indices[i+1]
                        if end - start > 100: # Filter tiny fragments
                            self.samples_info.append((file_path, start, end))
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")

        print(f"Found {len(self.samples_info)} scenes.")
        
        if not use_cache:
            self.use_cache = False
            return

        print("Caching scenes into RAM for Segmentation...")
        self.use_cache = True
        
        for file_path, start, end in tqdm(self.samples_info, desc="Caching"):
            with h5py.File(file_path, 'r') as f:
                ds = f['lidar_points']
                
                # Load Geometry
                dist = ds['distance_cm'][start:end]
                azim = ds['azimuth_raw'][start:end]
                elev = ds['elevation_raw'][start:end]
                
                # Load RGB Labels
                r = ds['r'][start:end]
                g = ds['g'][start:end]
                b = ds['b'][start:end]
                
                # Convert to XYZ
                xyz = lidar_utils.spherical_to_local_cartesian(dist, azim, elev)
                
                # Create Segmentation Mask (Default 4 = Background)
                seg_labels = np.full(r.shape, 4, dtype=np.int64)
                
                # Apply Masks based on RGB
                # Class 0: Antenna
                seg_labels[(r == 38) & (g == 23) & (b == 180)] = 0
                # Class 1: Cable
                seg_labels[(r == 177) & (g == 132) & (b == 47)] = 1
                # Class 2: Pole
                seg_labels[(r == 129) & (g == 81) & (b == 97)] = 2
                # Class 3: Turbine
                seg_labels[(r == 66) & (g == 132) & (b == 9)] = 3
                
                self.cached_samples.append((xyz, seg_labels))

    def __getitem__(self, idx):
        if self.use_cache:
            xyz, labels = self.cached_samples[idx]
        else:
            # Fallback not implemented for speed
            xyz, labels = self.cached_samples[idx]

        # Resamplings
        N = xyz.shape[0]
        if N >= self.n_points:
            choice = np.random.choice(N, self.n_points, replace=False)
        else:
            choice = np.random.choice(N, self.n_points, replace=True)

        xyz_sampled = xyz[choice, :]
        labels_sampled = labels[choice]

        # Normalization
        xyz_sampled = xyz_sampled - np.mean(xyz_sampled, axis=0) # Center
        max_dist = np.max(np.linalg.norm(xyz_sampled, axis=1))
        if max_dist > 0:
            xyz_sampled = xyz_sampled / max_dist

        # Return: (Channels, Points), (Points)
        # Input: (3, N)
        # Target: (N) - containing class ID per point
        return torch.from_numpy(xyz_sampled).float().transpose(1, 0), torch.from_numpy(labels_sampled).long()


if __name__ == "__main__":
    DATA_PATH = r"airbus_hackathon_trainingdata"
    dataset = LidarH5Dataset(DATA_PATH, n_points=1024)
    print(f"Dataset ready with {len(dataset)} samples.")
