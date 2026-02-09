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
        self.samples_info = [] # List of (file_path, start_idx, end_idx, label)
        
        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    ds = f['lidar_points']
                    N = ds.shape[0]
                    # We detect boundaries of poses by checking every 10k points 
                    # as an approximation, or just use the full pose array if it fits.
                    # Since we only need 4 columns, let's read them.
                    ego_x = ds['ego_x']
                    
                    # Simple boundary detection: where does ego_x change significantly?
                    # For this dataset, poses are usually contiguous blocks.
                    # Let's find unique segments.
                    step = 50000 # Typical number of points per scan is ~50k
                    for start in range(0, N, step):
                        end = min(start + step, N)
                        # Take a sample point from this block
                        p_idx = start + 10 # small offset
                        if p_idx >= N: p_idx = start
                        
                        # Calculate label for this block (approximate using middle point distance or full slice)
                        # For true accuracy, we'll do it in phase 2.
                        self.samples_info.append((file_path, start, end))
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")

        print(f"Found {len(self.samples_info)} potential scene blocks.")
        
        if not use_cache:
            self.use_cache = False
            return

        print("Caching scenes into RAM (Optimized Slicing)...")
        self.use_cache = True
        for file_path, start, end in tqdm(self.samples_info, desc="Caching"):
            with h5py.File(file_path, 'r') as f:
                ds = f['lidar_points']
                # Read ONLY the necessary columns for coordinates
                slice_data = ds[start:end]
                
                # Convert to Cartesian using logic from lidar_utils directly for speed
                dist_m = slice_data['distance_cm'] / 100.0
                azimuth_rad = np.radians(slice_data['azimuth_raw'] / 100.0)
                elevation_rad = np.radians(slice_data['elevation_raw'] / 100.0)
                
                x = dist_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
                y = -dist_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
                z = dist_m * np.sin(elevation_rad)
                xyz = np.column_stack((x, y, z)).astype(np.float32)
                
                # Labeling
                dist = np.linalg.norm(xyz, axis=1)
                angles = np.degrees(np.arctan2(y, x))
                mask_front = (np.abs(angles) < self.scan_angle_front) & (dist > 0.1)
                front_points_dist = dist[mask_front]
                
                label = 1 if (len(front_points_dist) > 0 and np.min(front_points_dist) < self.critical_dist) else 0
                self.cached_samples.append((xyz, label))


    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        xyz, label = self.cached_samples[idx]
        
        # Resampling to fixed N
        if len(xyz) >= self.n_points:
            choice = np.random.choice(len(xyz), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(xyz), self.n_points, replace=True)
        
        xyz_sampled = xyz[choice, :]
        
        # Normalize: zero-mean and unit sphere
        xyz_sampled = xyz_sampled - np.mean(xyz_sampled, axis=0)
        max_dist = np.max(np.sqrt(np.sum(xyz_sampled**2, axis=1)))
        if max_dist > 0:
            xyz_sampled = xyz_sampled / max_dist
            
        return torch.from_numpy(xyz_sampled.T), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    DATA_PATH = r"airbus_hackathon_trainingdata"
    dataset = LidarH5Dataset(DATA_PATH, n_points=1024)
    print(f"Dataset ready with {len(dataset)} samples.")
