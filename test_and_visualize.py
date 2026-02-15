"""
Combined test and visualization script for LiDAR obstacle detection.
Shows predictions vs ground truth side by side with metrics.

Usage:
    python test_and_visualize.py --checkpoint best.pth --file data/scene_1.h5 --pose-index 0
    python test_and_visualize.py --checkpoint best.pth --file data/scene_1.h5 --all
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

import lidar_utils
from pointnet_model import PointNetSegmentation
from dataset import NUM_CLASSES, CLASS_NAMES, rgb_array_to_class_ids


# ============================================
# Color mappings
# ============================================

# Colors for visualization (RGB normalized 0-1)
VIZ_COLORS = {
    0: [0.15, 0.09, 0.71],   # Antenna - Blue (same as GT)
    1: [0.69, 0.52, 0.18],   # Cable - Orange/Brown
    2: [0.51, 0.32, 0.38],   # Electric pole - Purple
    3: [0.26, 0.52, 0.04],   # Wind turbine - Green
    4: [0.5, 0.5, 0.5],      # Background - Gray
}

# More distinct colors for predictions
PRED_COLORS = {
    0: [0.0, 0.0, 1.0],      # Antenna - Bright Blue
    1: [1.0, 0.5, 0.0],      # Cable - Orange
    2: [0.8, 0.0, 0.8],      # Electric pole - Magenta
    3: [0.0, 1.0, 0.0],      # Wind turbine - Bright Green
    4: [0.3, 0.3, 0.3],      # Background - Dark Gray
}


# ============================================
# Model loading
# ============================================

"""def load_model(checkpoint_path, device):
    Load trained model from checkpoint
    print(f"Loading model from {checkpoint_path}...")

    # PyTorch 2.6+ compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect if enhanced decoder was used (check for conv0 layer)
    state_dict = checkpoint['model_state_dict']
    enhanced_decoder = 'conv0.weight' in state_dict

    model = PointNetSegmentation(
        num_classes=NUM_CLASSES,
        input_channels=4,
        enhanced_decoder=enhanced_decoder
    )
    print(f"  Enhanced decoder: {enhanced_decoder}")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Print info
    if 'epoch' in checkpoint:
        print(f"  Trained epochs: {checkpoint['epoch'] + 1}")
    if 'best_miou' in checkpoint:
        print(f"  Best mIoU: {checkpoint['best_miou']*100:.2f}%")
    if 'val_acc' in checkpoint:
        print(f"  Val Accuracy: {checkpoint['val_acc']*100:.2f}%")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    return model"""

def load_model(checkpoint_path, device):

    print(f"Loading model from {checkpoint_path}...")

    # PyTorch 2.6+ compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Détection du décodeur renforcé : chercher 'conv0.weight' dans les clés (même avec préfixe)
    state_dict = checkpoint['model_state_dict']
    enhanced_decoder = any('conv0.weight' in k for k in state_dict.keys())
    print(f"  Enhanced decoder: {enhanced_decoder}")

    # Créer le modèle avec la bonne architecture
    model = PointNetSegmentation(
        num_classes=NUM_CLASSES,
        input_channels=4,
        enhanced_decoder=enhanced_decoder
    )

    # Nettoyer les clés si elles viennent d'un modèle compilé (préfixe '_orig_mod.')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_key = k[10:]  # enlever '_orig_mod.'
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # Print info
    if 'epoch' in checkpoint:
        print(f"  Trained epochs: {checkpoint['epoch'] + 1}")
    if 'best_miou' in checkpoint:
        print(f"  Best mIoU: {checkpoint['best_miou']*100:.2f}%")
    if 'val_acc' in checkpoint:
        print(f"  Val Accuracy: {checkpoint['val_acc']*100:.2f}%")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    return model
# ============================================
# Inference
# ============================================

def normalize_points(xyz):
    """Normalize point cloud to unit sphere."""
    centroid = np.mean(xyz, axis=0)
    xyz_norm = xyz - centroid
    max_dist = np.max(np.sqrt(np.sum(xyz_norm ** 2, axis=1)))
    if max_dist > 0:
        xyz_norm = xyz_norm / max_dist
    return xyz_norm, centroid, max_dist


def predict_segmentation(model, xyz, reflectivity, device, batch_size=65536):
    """Run segmentation inference on point cloud."""
    n_points = len(xyz)

    # Normalize
    xyz_norm, centroid, scale = normalize_points(xyz)

    # Build features [x, y, z, reflectivity]
    if reflectivity is not None:
        features = np.column_stack([xyz_norm, reflectivity / 255.0])
    else:
        features = np.column_stack([xyz_norm, np.zeros(n_points)])

    features = features.astype(np.float32)

    # Process in batches
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch = features[i:i+batch_size]
            batch_len = len(batch)

            # Pad if needed (model expects fixed size for some operations)
            if batch_len < batch_size:
                pad = np.zeros((batch_size - batch_len, 4), dtype=np.float32)
                batch = np.vstack([batch, pad])

            tensor = torch.from_numpy(batch).unsqueeze(0).to(device)
            outputs, _, _ = model(tensor)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=2)
            preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()
            probs = probs.squeeze(0).cpu().numpy()

            # Remove padding
            all_preds.append(preds[:batch_len])
            all_probs.append(probs[:batch_len])

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)

    return predictions, probabilities


# ============================================
# Metrics
# ============================================

def compute_metrics(predictions, gt_labels):
    """Compute accuracy and IoU metrics."""
    # Overall accuracy
    accuracy = (predictions == gt_labels).mean()

    # Per-class metrics
    ious = {}
    precisions = {}
    recalls = {}

    for cls in range(NUM_CLASSES):
        pred_mask = predictions == cls
        gt_mask = gt_labels == cls

        tp = (pred_mask & gt_mask).sum()
        fp = (pred_mask & ~gt_mask).sum()
        fn = (~pred_mask & gt_mask).sum()

        # IoU
        union = (pred_mask | gt_mask).sum()
        ious[cls] = tp / union if union > 0 else float('nan')

        # Precision & Recall
        precisions[cls] = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

    # mIoU (obstacle classes only: 0-3)
    obstacle_ious = [ious[c] for c in range(4) if not np.isnan(ious[c])]
    miou = np.mean(obstacle_ious) if obstacle_ious else 0.0

    return {
        'accuracy': accuracy,
        'miou': miou,
        'ious': ious,
        'precisions': precisions,
        'recalls': recalls
    }


def print_metrics(metrics, title="Metrics"):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Mean IoU (obstacles): {metrics['miou']*100:.2f}%")
    print(f"\n{'Class':<20} {'IoU':>10} {'Precision':>12} {'Recall':>10}")
    print("-" * 55)

    for cls in range(NUM_CLASSES):
        iou = metrics['ious'][cls]
        prec = metrics['precisions'][cls]
        rec = metrics['recalls'][cls]

        iou_str = f"{iou*100:.2f}%" if not np.isnan(iou) else "N/A"
        prec_str = f"{prec*100:.2f}%" if not np.isnan(prec) else "N/A"
        rec_str = f"{rec*100:.2f}%" if not np.isnan(rec) else "N/A"

        print(f"{CLASS_NAMES[cls]:<20} {iou_str:>10} {prec_str:>12} {rec_str:>10}")


def print_class_distribution(predictions, gt_labels=None):
    """Print class distribution comparison."""
    print(f"\n{'Class':<20} {'Predicted':>12} {'Ground Truth':>14} {'Diff':>10}")
    print("-" * 58)

    for cls in range(NUM_CLASSES):
        pred_count = (predictions == cls).sum()
        pred_pct = 100 * pred_count / len(predictions)

        if gt_labels is not None:
            gt_count = (gt_labels == cls).sum()
            gt_pct = 100 * gt_count / len(gt_labels)
            diff = pred_count - gt_count
            diff_str = f"{diff:+d}"
        else:
            gt_count = "N/A"
            gt_pct = 0
            diff_str = "-"

        print(f"{CLASS_NAMES[cls]:<20} {pred_count:>8} ({pred_pct:>4.1f}%) {gt_count:>8} ({gt_pct:>4.1f}%) {diff_str:>10}")


# ============================================
# Visualization
# ============================================

def visualize_comparison(xyz, predictions, gt_labels=None, probabilities=None, title="Predictions"):
    """Visualize predictions vs ground truth side by side."""
    try:
        import open3d as o3d
    except ImportError:
        print("\nOpen3D not installed. Run: pip install open3d")
        print("Skipping visualization.")
        return

    geometries = []

    # Calculate scene bounds for positioning
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    offset = max(x_range, y_range) * 1.2

    # 1. PREDICTIONS (left side)
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(xyz)
    colors_pred = np.array([PRED_COLORS[p] for p in predictions])
    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
    geometries.append(pcd_pred)

    # 2. GROUND TRUTH (right side, shifted)
    if gt_labels is not None:
        pcd_gt = o3d.geometry.PointCloud()
        xyz_gt = xyz.copy()
        xyz_gt[:, 1] += offset  # Shift along Y axis
        pcd_gt.points = o3d.utility.Vector3dVector(xyz_gt)
        colors_gt = np.array([PRED_COLORS[g] for g in gt_labels])
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)
        geometries.append(pcd_gt)

    # 3. ERROR VISUALIZATION (far right, only wrong predictions)
    if gt_labels is not None:
        errors = predictions != gt_labels
        if errors.sum() > 0:
            pcd_err = o3d.geometry.PointCloud()
            xyz_err = xyz.copy()
            xyz_err[:, 1] += offset * 2

            # Color: red for errors, very transparent gray for correct
            colors_err = np.zeros((len(xyz), 3))
            colors_err[errors] = [1.0, 0.0, 0.0]  # Red for errors
            colors_err[~errors] = [0.2, 0.2, 0.2]  # Dark gray for correct

            pcd_err.points = o3d.utility.Vector3dVector(xyz_err)
            pcd_err.colors = o3d.utility.Vector3dVector(colors_err)
            geometries.append(pcd_err)

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    geometries.append(coord_frame)

    # Print legend
    print("\n" + "="*60)
    print("VISUALIZATION LEGEND")
    print("="*60)
    if gt_labels is not None:
        print("Left:   PREDICTIONS")
        print("Center: GROUND TRUTH")
        print("Right:  ERRORS (red = wrong, gray = correct)")
    else:
        print("Showing: PREDICTIONS")

    print("\nColor coding:")
    for cls, color in PRED_COLORS.items():
        r, g, b = [int(c*255) for c in color]
        print(f"  {CLASS_NAMES[cls]:<15}: RGB({r:3d}, {g:3d}, {b:3d})")

    print("\nControls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  Shift+drag: Pan")
    print("  Q: Quit")

    # Launch viewer
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1600,
        height=900,
        point_show_normal=False
    )


# ============================================
# Main
# ============================================

def process_single_frame(model, df, pose, device):
    """Process a single frame and return results."""
    # Filter by pose
    frame_df = lidar_utils.filter_by_pose(df, pose)

    # Filter valid points
    valid_mask = frame_df['distance_cm'] > 0
    frame_df = frame_df[valid_mask].reset_index(drop=True)

    if len(frame_df) == 0:
        return None

    # Convert to Cartesian
    xyz = lidar_utils.spherical_to_local_cartesian(frame_df).astype(np.float32)

    # Get reflectivity
    reflectivity = None
    if 'reflectivity' in frame_df.columns:
        reflectivity = frame_df['reflectivity'].values

    # Get ground truth labels
    gt_labels = None
    if {'r', 'g', 'b'}.issubset(frame_df.columns):
        rgb = frame_df[['r', 'g', 'b']].values
        # Check if real labels (not all 128)
        if not (rgb[:, 0] == 128).all():
            gt_labels = rgb_array_to_class_ids(rgb)

    # Run prediction
    predictions, probabilities = predict_segmentation(model, xyz, reflectivity, device)

    # Compute metrics
    metrics = None
    if gt_labels is not None:
        metrics = compute_metrics(predictions, gt_labels)

    return {
        'xyz': xyz,
        'predictions': predictions,
        'probabilities': probabilities,
        'gt_labels': gt_labels,
        'metrics': metrics,
        'pose': pose.to_dict(),
        'n_points': len(xyz)
    }


def main():
    parser = argparse.ArgumentParser(description="Test and visualize LiDAR model predictions")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (best.pth)")
    parser.add_argument("--file", required=True, help="Path to HDF5 data file")
    parser.add_argument("--pose-index", type=int, default=None, help="Specific pose to test")
    parser.add_argument("--all", action="store_true", help="Test all poses in file")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load data
    print(f"\nLoading {args.file}...")
    df = lidar_utils.load_h5_data(args.file)
    pose_counts = lidar_utils.get_unique_poses(df)
    print(f"Found {len(pose_counts)} unique poses, {len(df)} total points")

    # Determine which poses to process
    if args.pose_index is not None:
        if args.pose_index < 0 or args.pose_index >= len(pose_counts):
            print(f"Invalid pose index {args.pose_index}. Valid range: 0-{len(pose_counts)-1}")
            return
        pose_indices = [args.pose_index]
    elif args.all:
        pose_indices = list(range(len(pose_counts)))
    else:
        # Show available poses
        print("\nAvailable poses:")
        print(pose_counts[["pose_index", "ego_x", "ego_y", "ego_z", "ego_yaw", "num_points"]].to_string(index=False))
        print("\nUsage:")
        print("  --pose-index N  : Test specific pose")
        print("  --all           : Test all poses")
        return

    # Process frames
    all_results = []

    print(f"\nProcessing {len(pose_indices)} frame(s)...")
    for pose_idx in tqdm(pose_indices, desc="Frames"):
        pose = pose_counts.iloc[pose_idx]
        result = process_single_frame(model, df, pose, device)

        if result is not None:
            result['pose_index'] = pose_idx
            all_results.append(result)

    # Single frame: detailed output + visualization
    if len(all_results) == 1:
        result = all_results[0]

        print(f"\nPose {result['pose_index']}: {result['n_points']} points")

        if result['metrics']:
            print_metrics(result['metrics'], f"Frame {result['pose_index']} Metrics")

        print_class_distribution(result['predictions'], result['gt_labels'])

        if not args.no_viz:
            visualize_comparison(
                result['xyz'],
                result['predictions'],
                result['gt_labels'],
                result['probabilities'],
                title=f"Pose {result['pose_index']} - mIoU: {result['metrics']['miou']*100:.1f}%" if result['metrics'] else f"Pose {result['pose_index']}"
            )

    # Multiple frames: aggregate metrics
    elif len(all_results) > 1:
        print("\n" + "="*60)
        print("AGGREGATE RESULTS")
        print("="*60)

        # Collect metrics
        valid_results = [r for r in all_results if r['metrics'] is not None]

        if valid_results:
            avg_acc = np.mean([r['metrics']['accuracy'] for r in valid_results])
            avg_miou = np.mean([r['metrics']['miou'] for r in valid_results])

            print(f"Frames tested: {len(valid_results)}")
            print(f"Average Accuracy: {avg_acc*100:.2f}%")
            print(f"Average mIoU: {avg_miou*100:.2f}%")

            print(f"\n{'Class':<20} {'Avg IoU':>10} {'Avg Precision':>14} {'Avg Recall':>12}")
            print("-" * 58)

            for cls in range(NUM_CLASSES):
                ious = [r['metrics']['ious'][cls] for r in valid_results if not np.isnan(r['metrics']['ious'][cls])]
                precs = [r['metrics']['precisions'][cls] for r in valid_results if not np.isnan(r['metrics']['precisions'][cls])]
                recs = [r['metrics']['recalls'][cls] for r in valid_results if not np.isnan(r['metrics']['recalls'][cls])]

                iou_str = f"{np.mean(ious)*100:.2f}%" if ious else "N/A"
                prec_str = f"{np.mean(precs)*100:.2f}%" if precs else "N/A"
                rec_str = f"{np.mean(recs)*100:.2f}%" if recs else "N/A"

                print(f"{CLASS_NAMES[cls]:<20} {iou_str:>10} {prec_str:>14} {rec_str:>12}")

            # Find best and worst frames
            sorted_results = sorted(valid_results, key=lambda x: x['metrics']['miou'], reverse=True)

            print(f"\nBest frame:  Pose {sorted_results[0]['pose_index']} (mIoU: {sorted_results[0]['metrics']['miou']*100:.2f}%)")
            print(f"Worst frame: Pose {sorted_results[-1]['pose_index']} (mIoU: {sorted_results[-1]['metrics']['miou']*100:.2f}%)")

        else:
            print("No ground truth labels found in data.")

    print("\nDone!")


if __name__ == "__main__":
    main()
