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
        # Persistent cache path
        cache_path = os.path.join(self.data_dir, f"cache_seg_{self.n_points}.pt")
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading persistent cache from {cache_path}...")
            self.cached_samples = torch.load(cache_path)
            self.use_cache = True
            print(f"Loaded {len(self.cached_samples)} cached scenes.")
            return

        print(f"Indexing {len(self.files)} H5 files efficiently...")
        self.samples_info = [] 

        for file_path in self.files:
            try:
                with h5py.File(file_path, 'r') as f:
                    ds = f['lidar_points']
                    N = ds.shape[0]
                    ego_x = ds['ego_x'][:]
                    diff = np.abs(np.diff(ego_x))
                    change_indices = np.where(diff > 100)[0] + 1
                    indices = np.concatenate(([0], change_indices, [N]))
                    for i in range(len(indices) - 1):
                        start = indices[i]
                        end = indices[i+1]
                        if end - start > 100:
                            self.samples_info.append((file_path, start, end))
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")

        print(f"Found {len(self.samples_info)} scenes.")
        
        if not use_cache:
            self.use_cache = False
            return

        print("Caching scenes into RAM (Surgical Sampling)...")
        self.use_cache = True
        self.cached_samples = [None] * len(self.samples_info)
        
        from collections import defaultdict
        file_map = defaultdict(list)
        for idx, (path, start, end) in enumerate(self.samples_info):
            file_map[path].append((idx, start, end))
            
        for file_path, scenes in tqdm(file_map.items(), desc="Caching Files"):
            try:
                with h5py.File(file_path, 'r') as f:
                    ds = f['lidar_points']
                    for idx, start, end in scenes:
                        # CONTIGUOUS READ: Much faster than indexed reads
                        scene_data = ds[start:end]
                        n_in_scene = len(scene_data)
                        
                        # Sample indices in RAM
                        if n_in_scene >= self.n_points:
                            choice = np.random.choice(n_in_scene, self.n_points, replace=False)
                        else:
                            choice = np.random.choice(n_in_scene, self.n_points, replace=True)
                        
                        # Extract sampled data
                        sampled = scene_data[choice]
                        
                        data_dict = {
                            "distance_cm": sampled['distance_cm'],
                            "azimuth_raw": sampled['azimuth_raw'],
                            "elevation_raw": sampled['elevation_raw']
                        }
                        xyz = lidar_utils.spherical_to_local_cartesian(data_dict)
                        
                        r, g, b = sampled['r'], sampled['g'], sampled['b']
                        
                        seg_labels = np.full(r.shape, 4, dtype=np.int64)
                        seg_labels[(r == 38) & (g == 23) & (b == 180)] = 0
                        seg_labels[(r == 177) & (g == 132) & (b == 47)] = 1
                        seg_labels[(r == 129) & (g == 81) & (b == 97)] = 2
                        seg_labels[(r == 66) & (g == 132) & (b == 9)] = 3
                        
                        self.cached_samples[idx] = (xyz, seg_labels)
            except Exception as e:
                print(f"Error caching {file_path}: {e}")

        self.cached_samples = [s for s in self.cached_samples if s is not None]
        
        # Save cache for next time
        print(f"Saving cache to {cache_path}...")
        torch.save(self.cached_samples, cache_path)

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        # Data is already sampled and ready in RAM!
        xyz, labels = self.cached_samples[idx]

        # Final normalization (Center & Scale)
        xyz_norm = xyz - np.mean(xyz, axis=0) # Center
        max_dist = np.max(np.linalg.norm(xyz_norm, axis=1))
        if max_dist > 0:
            xyz_norm = xyz_norm / max_dist

        # Transpose for PyTorch (3, N)
        return torch.from_numpy(xyz_norm).float().transpose(1, 0), torch.from_numpy(labels).long()


if __name__ == "__main__":
    DATA_PATH = r"airbus_hackathon_trainingdata"
    dataset = LidarH5Dataset(DATA_PATH, n_points=1024)
    print(f"Dataset ready with {len(dataset)} samples.")
