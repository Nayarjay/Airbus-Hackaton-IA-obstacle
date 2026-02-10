import os
import numpy as np
import torch
from torch.utils.data import Dataset
import lidar_utils


# Class color mapping (RGB values)
CLASS_COLORS = {
    0: (38, 23, 180),      # Antenna
    1: (177, 132, 47),     # Cable
    2: (129, 81, 97),      # Electric pole
    3: (66, 132, 9),       # Wind turbine
}

CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric pole",
    3: "Wind turbine",
    4: "Background"
}

NUM_CLASSES = 5  # 4 obstacles + 1 background


def rgb_to_class_id(r, g, b):
    """Map RGB tuple to class ID."""
    for class_id, (cr, cg, cb) in CLASS_COLORS.items():
        if r == cr and g == cg and b == cb:
            return class_id
    return 4  # Background


def rgb_array_to_class_ids(rgb_array):
    """Convert array of RGB values to class IDs."""
    r, g, b = rgb_array[:, 0], rgb_array[:, 1], rgb_array[:, 2]
    class_ids = np.full(len(rgb_array), 4, dtype=np.int64)  # Default: background

    for class_id, (cr, cg, cb) in CLASS_COLORS.items():
        mask = (r == cr) & (g == cg) & (b == cb)
        class_ids[mask] = class_id

    return class_ids


class LidarDataset(Dataset):
    """PyTorch Dataset for LiDAR point cloud segmentation."""

    def __init__(
        self,
        data_dir,
        n_points=32768,
        augment=True,
        normalize=True,
        use_reflectivity=True
    ):
        """
        Args:
            data_dir: Directory containing HDF5 files
            n_points: Number of points to sample per frame
            augment: Whether to apply data augmentation
            normalize: Whether to normalize coordinates
            use_reflectivity: Whether to include reflectivity as feature
        """
        self.data_dir = data_dir
        self.n_points = n_points
        self.augment = augment
        self.normalize = normalize
        self.use_reflectivity = use_reflectivity

        # Load all frames from all HDF5 files
        self.frames = []
        self._load_all_frames()

    def _load_all_frames(self):
        """Load metadata for all frames from all HDF5 files."""
        h5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]

        for h5_file in sorted(h5_files):
            file_path = os.path.join(self.data_dir, h5_file)
            try:
                df = lidar_utils.load_h5_data(file_path)
                pose_counts = lidar_utils.get_unique_poses(df)

                if pose_counts is not None:
                    for idx in range(len(pose_counts)):
                        self.frames.append({
                            'file_path': file_path,
                            'pose_index': idx,
                            'pose_data': pose_counts.iloc[idx].to_dict()
                        })
            except Exception as e:
                print(f"Warning: Could not load {h5_file}: {e}")

        print(f"Loaded {len(self.frames)} frames from {len(h5_files)} files")

    def __len__(self):
        return len(self.frames)

    def _augment_points(self, xyz):
        """Apply data augmentation to point cloud."""
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        xyz = xyz @ rotation_matrix.T

        # Random jitter
        jitter = np.random.normal(0, 0.01, size=xyz.shape)
        xyz = xyz + jitter

        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        xyz = xyz * scale

        return xyz

    def _normalize_points(self, xyz):
        """Normalize point cloud to unit sphere."""
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if max_dist > 0:
            xyz = xyz / max_dist
        return xyz

    def _sample_points(self, xyz, features, labels):
        """Sample or pad point cloud to fixed size."""
        n_points_current = xyz.shape[0]

        # Handle empty point clouds
        if n_points_current == 0:
            xyz = np.zeros((self.n_points, 3), dtype=np.float32)
            features = np.zeros(self.n_points, dtype=np.float32) if features is not None else None
            labels = np.full(self.n_points, 4, dtype=np.int64)  # Background class
            return xyz, features, labels

        if n_points_current >= self.n_points:
            # Random sampling
            indices = np.random.choice(n_points_current, self.n_points, replace=False)
        else:
            # Pad with duplicates
            indices = np.random.choice(n_points_current, self.n_points, replace=True)

        xyz = xyz[indices]
        features = features[indices] if features is not None else None
        labels = labels[indices]

        return xyz, features, labels

    def __getitem__(self, idx):
        frame_info = self.frames[idx]

        # Load data
        df = lidar_utils.load_h5_data(frame_info['file_path'])
        pose_counts = lidar_utils.get_unique_poses(df)
        selected_pose = pose_counts.iloc[frame_info['pose_index']]
        frame_df = lidar_utils.filter_by_pose(df, selected_pose)

        # Filter valid points
        valid_mask = frame_df['distance_cm'] > 0
        frame_df = frame_df[valid_mask].reset_index(drop=True)

        # Convert to Cartesian coordinates
        xyz = lidar_utils.spherical_to_local_cartesian(frame_df)

        # Extract labels
        rgb_values = frame_df[['r', 'g', 'b']].values
        labels = rgb_array_to_class_ids(rgb_values)

        # Extract reflectivity
        reflectivity = None
        if self.use_reflectivity and 'reflectivity' in frame_df.columns:
            reflectivity = frame_df['reflectivity'].values.astype(np.float32) / 255.0

        # Sample/pad to fixed size
        xyz, reflectivity, labels = self._sample_points(xyz, reflectivity, labels)

        # Normalize coordinates
        if self.normalize:
            xyz = self._normalize_points(xyz)

        # Apply augmentation
        if self.augment:
            xyz = self._augment_points(xyz)

        # Build feature tensor
        xyz = xyz.astype(np.float32)
        if self.use_reflectivity and reflectivity is not None:
            features = np.column_stack([xyz, reflectivity])
        else:
            features = xyz

        # Convert to tensors
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        return features, labels

    def get_class_weights(self):
        """Compute class weights based on inverse frequency."""
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)

        # Sample a subset of frames for efficiency
        sample_size = min(20, len(self.frames))
        sample_indices = np.random.choice(len(self.frames), sample_size, replace=False)

        for idx in sample_indices:
            _, labels = self[idx]
            for c in range(NUM_CLASSES):
                class_counts[c] += (labels == c).sum().item()

        # Compute inverse frequency weights
        total = class_counts.sum()
        weights = total / (NUM_CLASSES * class_counts + 1e-6)

        # Normalize
        weights = weights / weights.sum() * NUM_CLASSES

        return torch.from_numpy(weights).float()


def get_dataloaders(data_dir, batch_size=8, n_points=32768, num_workers=4, val_split=0.2):
    """Create train and validation dataloaders."""
    full_dataset = LidarDataset(
        data_dir=data_dir,
        n_points=n_points,
        augment=True,
        normalize=True,
        use_reflectivity=True
    )

    # Split into train and validation
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Create validation dataset without augmentation
    val_dataset_no_aug = LidarDataset(
        data_dir=data_dir,
        n_points=n_points,
        augment=False,
        normalize=True,
        use_reflectivity=True
    )

    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, full_dataset.get_class_weights()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]

    dataset = LidarDataset(data_dir, n_points=4096, augment=False)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        features, labels = dataset[0]
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels unique: {torch.unique(labels)}")

        # Class distribution
        for c in range(NUM_CLASSES):
            count = (labels == c).sum().item()
            print(f"  Class {c} ({CLASS_NAMES[c]}): {count} points ({100*count/len(labels):.1f}%)")

        weights = dataset.get_class_weights()
        print(f"Class weights: {weights}")
