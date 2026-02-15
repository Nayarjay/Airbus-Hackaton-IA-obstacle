import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import lidar_utils
from pointnet_model import PointNetSegmentation
from dataset import NUM_CLASSES, CLASS_NAMES, rgb_array_to_class_ids


def normalize_points(xyz):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(xyz, axis=0)
    xyz_normalized = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz_normalized ** 2, axis=1)))
    if max_dist > 0:
        xyz_normalized = xyz_normalized / max_dist
    return xyz_normalized, centroid, max_dist


def compute_oriented_bbox(points):
    """Compute oriented bounding box using PCA."""
    if len(points) < 3:
        center = points.mean(axis=0)
        return center, np.array([0.1, 0.1, 0.1]), 0.0

    # Center the points
    center = points.mean(axis=0)
    centered = points - center

    # PCA for orientation (2D on XY plane)
    xy = centered[:, :2]
    if xy.shape[0] > 1:
        cov = np.cov(xy.T)
        if cov.ndim == 2:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Principal direction
            principal_dir = eigenvectors[:, np.argmax(eigenvalues)]
            yaw = np.arctan2(principal_dir[1], principal_dir[0])
        else:
            yaw = 0.0
    else:
        yaw = 0.0

    # Rotate points to align with axes
    cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    rotated = centered @ rotation_matrix.T

    # Compute axis-aligned dimensions
    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    dimensions = maxs - mins

    # Ensure minimum dimensions
    dimensions = np.maximum(dimensions, 0.1)

    return center, dimensions, yaw


def cluster_and_bbox(points, class_id, eps=2.0, min_samples=10):
    """Cluster points and compute bounding boxes for each cluster."""
    if len(points) < min_samples:
        return []

    # Adjust clustering parameters based on class
    if class_id == 1:  # Cable - thin, elongated
        eps = 1.5
        min_samples = 5
    elif class_id == 3:  # Wind turbine - large
        eps = 5.0
        min_samples = 50
    elif class_id == 0:  # Antenna
        eps = 2.0
        min_samples = 15
    elif class_id == 2:  # Electric pole
        eps = 2.5
        min_samples = 20

    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    bboxes = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = points[cluster_mask]

        if len(cluster_points) >= min_samples:
            center, dimensions, yaw = compute_oriented_bbox(cluster_points)
            bboxes.append({
                'center': center,
                'dimensions': dimensions,
                'yaw': yaw,
                'num_points': len(cluster_points)
            })

    return bboxes


def predict_frame(model, xyz, reflectivity, device, batch_size=65536):
    """Predict segmentation for a frame (handles large point clouds)."""
    model.eval()
    n_points = len(xyz)

    # Normalize
    xyz_norm, centroid, scale = normalize_points(xyz)

    # Build features
    if reflectivity is not None:
        features = np.column_stack([xyz_norm, reflectivity / 255.0])
    else:
        features = np.column_stack([xyz_norm, np.zeros(n_points)])

    features = features.astype(np.float32)

    # Process in batches if too large
    all_preds = []

    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_features = features[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch_features).unsqueeze(0).to(device)

            outputs, _, _ = model(batch_tensor)
            preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()
            all_preds.append(preds)

    predictions = np.concatenate(all_preds)
    return predictions


def process_frame(model, frame_df, pose_data, device):
    """Process a single frame and generate bounding boxes."""
    # Filter valid points
    valid_mask = frame_df['distance_cm'] > 0
    valid_df = frame_df[valid_mask].reset_index(drop=True)

    if len(valid_df) == 0:
        return []

    # Convert to Cartesian (meters)
    xyz = lidar_utils.spherical_to_local_cartesian(valid_df)

    # Get reflectivity
    reflectivity = None
    if 'reflectivity' in valid_df.columns:
        reflectivity = valid_df['reflectivity'].values

    # Predict segmentation
    predictions = predict_frame(model, xyz, reflectivity, device)

    # Generate bounding boxes for each obstacle class
    results = []

    for class_id in range(4):  # Only obstacle classes (0-3)
        class_mask = predictions == class_id
        class_points = xyz[class_mask]

        if len(class_points) > 0:
            bboxes = cluster_and_bbox(class_points, class_id)

            for bbox in bboxes:
                results.append({
                    'ego_x': pose_data['ego_x'],
                    'ego_y': pose_data['ego_y'],
                    'ego_z': pose_data['ego_z'],
                    'ego_yaw': pose_data['ego_yaw'],
                    'bbox_center_x': bbox['center'][0],
                    'bbox_center_y': bbox['center'][1],
                    'bbox_center_z': bbox['center'][2],
                    'bbox_width': bbox['dimensions'][0],
                    'bbox_length': bbox['dimensions'][1],
                    'bbox_height': bbox['dimensions'][2],
                    'bbox_yaw': bbox['yaw'],
                    'Class ID': class_id,
                    'Class Label': CLASS_NAMES[class_id]
                })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with PointNet Segmentation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input HDF5 file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Strip _orig_mod. prefix if model was saved with torch.compile()
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Detect if enhanced decoder was used
    enhanced_decoder = 'conv0.weight' in state_dict
    print(f"  Enhanced decoder: {enhanced_decoder}")

    model = PointNetSegmentation(
        num_classes=NUM_CLASSES,
        input_channels=4,
        enhanced_decoder=enhanced_decoder
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Print model info
    if 'best_miou' in checkpoint:
        print(f"  Best mIoU: {checkpoint['best_miou']*100:.2f}%")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Collect input files
    if os.path.isdir(args.input):
        h5_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.h5')]
    else:
        h5_files = [args.input]

    print(f"Processing {len(h5_files)} file(s)")

    # Process all files
    all_results = []

    for h5_file in tqdm(h5_files, desc="Files"):
        print(f"\nProcessing {h5_file}")

        try:
            df = lidar_utils.load_h5_data(h5_file)
            pose_counts = lidar_utils.get_unique_poses(df)

            if pose_counts is None:
                print(f"  No pose data found, skipping")
                continue

            for pose_idx in tqdm(range(len(pose_counts)), desc="Frames", leave=False):
                pose_data = pose_counts.iloc[pose_idx].to_dict()
                frame_df = lidar_utils.filter_by_pose(df, pose_counts.iloc[pose_idx])

                results = process_frame(model, frame_df, pose_data, device)
                all_results.extend(results)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)

        # Reorder columns as per specification
        column_order = [
            'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
            'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
            'bbox_width', 'bbox_length', 'bbox_height',
            'bbox_yaw', 'Class ID', 'Class Label'
        ]
        results_df = results_df[column_order]

        results_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(results_df)} predictions to {args.output}")

        # Summary
        print("\nPrediction summary:")
        for class_id in range(4):
            count = (results_df['Class ID'] == class_id).sum()
            print(f"  {CLASS_NAMES[class_id]}: {count} detections")
    else:
        print("\nNo detections found")
        pd.DataFrame(columns=[
            'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
            'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
            'bbox_width', 'bbox_length', 'bbox_height',
            'bbox_yaw', 'Class ID', 'Class Label'
        ]).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
