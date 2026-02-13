"""
Simple test script for the trained model.
Visualizes predictions vs ground truth.

Usage:
    python test_model.py --checkpoint best.pth --file data/scene_1.h5 --pose-index 0
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

import lidar_utils
from pointnet_model import PointNetSegmentation
from dataset import NUM_CLASSES, CLASS_NAMES, rgb_array_to_class_ids

# Prediction colors (RGB normalized)
PRED_COLORS = {
    0: [0.0, 0.0, 1.0],      # Antenna - Blue
    1: [1.0, 0.65, 0.0],     # Cable - Orange
    2: [0.5, 0.0, 0.5],      # Electric pole - Purple
    3: [0.0, 1.0, 0.0],      # Wind turbine - Green
    4: [0.4, 0.4, 0.4],      # Background - Gray
}


def load_model(checkpoint_path, device):
    """Load the trained model."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = PointNetSegmentation(num_classes=NUM_CLASSES, input_channels=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    if 'best_miou' in checkpoint:
        print(f"  Best mIoU: {checkpoint['best_miou']*100:.2f}%")
    if 'val_acc' in checkpoint:
        print(f"  Val Accuracy: {checkpoint['val_acc']*100:.2f}%")

    return model


def predict_frame(model, xyz, reflectivity, device, n_points=65536):
    """Run inference on a point cloud frame."""
    n_total = len(xyz)

    # Normalize
    centroid = np.mean(xyz, axis=0)
    xyz_norm = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz_norm ** 2, axis=1)))
    if max_dist > 0:
        xyz_norm = xyz_norm / max_dist

    # Build features
    if reflectivity is not None:
        features = np.column_stack([xyz_norm, reflectivity / 255.0]).astype(np.float32)
    else:
        features = np.column_stack([xyz_norm, np.zeros(n_total)]).astype(np.float32)

    # Process in chunks if needed
    all_preds = []

    with torch.no_grad():
        for i in range(0, n_total, n_points):
            chunk = features[i:i+n_points]

            # Pad if needed
            if len(chunk) < n_points:
                pad_size = n_points - len(chunk)
                chunk = np.vstack([chunk, np.zeros((pad_size, 4), dtype=np.float32)])

            tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
            outputs, _, _ = model(tensor)
            preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()

            # Remove padding
            actual_size = min(n_points, n_total - i)
            all_preds.append(preds[:actual_size])

    return np.concatenate(all_preds)


def compute_metrics(predictions, gt_labels):
    """Compute IoU and accuracy."""
    accuracy = (predictions == gt_labels).mean()

    ious = {}
    for cls in range(NUM_CLASSES):
        pred_mask = predictions == cls
        gt_mask = gt_labels == cls
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        ious[cls] = intersection / union if union > 0 else float('nan')

    obstacle_ious = [ious[c] for c in range(4) if not np.isnan(ious[c])]
    miou = np.mean(obstacle_ious) if obstacle_ious else 0.0

    return accuracy, ious, miou


def visualize(xyz, predictions, gt_labels=None):
    """Visualize point cloud with predictions."""
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Run: pip install open3d")
        return

    # Create prediction point cloud
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(xyz)

    colors_pred = np.array([PRED_COLORS[p] for p in predictions])
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

    geometries = [pcd_pred]

    # Create ground truth point cloud (shifted)
    if gt_labels is not None:
        pcd_gt = o3d.geometry.PointCloud()
        xyz_shifted = xyz.copy()
        xyz_shifted[:, 1] += (xyz[:, 1].max() - xyz[:, 1].min()) * 1.5
        pcd_gt.points = o3d.utility.Vector3dVector(xyz_shifted)

        colors_gt = np.array([PRED_COLORS[g] for g in gt_labels])
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
        geometries.append(pcd_gt)

        print("\nVisualization: Left = Predictions, Right = Ground Truth")

    # Legend
    print("\nColor legend:")
    for cls, color in PRED_COLORS.items():
        print(f"  {CLASS_NAMES[cls]}: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")

    o3d.visualization.draw_geometries(geometries, window_name="Model Predictions", width=1400, height=800)


def main():
    parser = argparse.ArgumentParser(description="Test trained LiDAR model")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pth")
    parser.add_argument("--file", required=True, help="Path to HDF5 file")
    parser.add_argument("--pose-index", type=int, default=None, help="Pose index to test")
    parser.add_argument("--all", action="store_true", help="Test all poses")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    print(f"\nLoading {args.file}...")
    df = lidar_utils.load_h5_data(args.file)
    pose_counts = lidar_utils.get_unique_poses(df)
    print(f"Found {len(pose_counts)} poses")

    # Determine poses to test
    if args.pose_index is not None:
        pose_indices = [args.pose_index]
    elif args.all:
        pose_indices = list(range(len(pose_counts)))
    else:
        print("\nAvailable poses:")
        print(pose_counts[["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"]].to_string(index=False))
        print("\nUse --pose-index N to test a specific pose, or --all for all poses")
        return

    # Test each pose
    all_metrics = []

    for pose_idx in tqdm(pose_indices, desc="Testing"):
        pose = pose_counts.iloc[pose_idx]
        frame_df = lidar_utils.filter_by_pose(df, pose)

        # Filter valid points
        valid_mask = frame_df['distance_cm'] > 0
        frame_df = frame_df[valid_mask].reset_index(drop=True)

        # Convert to Cartesian
        xyz = lidar_utils.spherical_to_local_cartesian(frame_df).astype(np.float32)

        # Get reflectivity
        reflectivity = frame_df['reflectivity'].values if 'reflectivity' in frame_df.columns else None

        # Get ground truth
        gt_labels = None
        if {'r', 'g', 'b'}.issubset(frame_df.columns):
            rgb = frame_df[['r', 'g', 'b']].values
            if not (rgb[:, 0] == 128).all():  # Check if labels exist
                gt_labels = rgb_array_to_class_ids(rgb)

        # Run prediction
        predictions = predict_frame(model, xyz, reflectivity, device)

        # Compute metrics
        if gt_labels is not None:
            acc, ious, miou = compute_metrics(predictions, gt_labels)
            all_metrics.append({'pose': pose_idx, 'acc': acc, 'miou': miou, 'ious': ious})

            print(f"\nPose {pose_idx}: Acc={acc*100:.2f}%, mIoU={miou*100:.2f}%")
            for cls in range(4):
                if not np.isnan(ious[cls]):
                    print(f"  {CLASS_NAMES[cls]}: {ious[cls]*100:.2f}%")

        # Visualize (single pose only)
        if len(pose_indices) == 1 and not args.no_viz:
            print("\nClass distribution in predictions:")
            for cls in range(NUM_CLASSES):
                count = (predictions == cls).sum()
                pct = 100 * count / len(predictions)
                print(f"  {CLASS_NAMES[cls]}: {count} ({pct:.1f}%)")

            visualize(xyz, predictions, gt_labels)

    # Summary for multiple poses
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        avg_acc = np.mean([m['acc'] for m in all_metrics])
        avg_miou = np.mean([m['miou'] for m in all_metrics])
        print(f"Average Accuracy: {avg_acc*100:.2f}%")
        print(f"Average mIoU: {avg_miou*100:.2f}%")

        print("\nPer-class average IoU:")
        for cls in range(NUM_CLASSES):
            ious = [m['ious'][cls] for m in all_metrics if not np.isnan(m['ious'][cls])]
            if ious:
                print(f"  {CLASS_NAMES[cls]}: {np.mean(ious)*100:.2f}%")


if __name__ == "__main__":
    main()
