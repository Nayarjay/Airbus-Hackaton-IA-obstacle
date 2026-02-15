import argparse
import numpy as np
import open3d as o3d
import torch

from src.airbus_lidar.io.h5_index import H5FrameIndex
from src.airbus_lidar.io.h5_reader import read_frame_fields
from src.airbus_lidar.geometry.coords import spherical_to_local_cartesian_np
from src.airbus_lidar.geometry.bbox import cluster_and_build_bboxes
from src.airbus_lidar.data.dataset import rgb_to_class_id
from src.airbus_lidar.constants import BACKGROUND_ID, NUM_CLASSES, RGB_TO_CLASS_ID
from src.airbus_lidar.models.pointnet_seg import PointNetSeg
from src.airbus_lidar.infer.inference import load_model
from src.airbus_lidar.config import DataConfig, InferConfig

# Couleurs des boxes = couleurs GT (RGB dataset)
CLASS_ID_TO_RGB = {
    cid: np.array(rgb, dtype=np.float32) / 255.0
    for rgb, cid in RGB_TO_CLASS_ID.items()
}


def yaw_to_R(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .h5")
    parser.add_argument("--pose-index", type=int, required=True, help="Frame index (0-based)")
    parser.add_argument("--mode", choices=["gt", "pred"], default="gt", help="gt=labels RGB, pred=model")
    parser.add_argument("--checkpoint", default="checkpoints/pointnet_seg_best.pt", help="Used only if mode=pred")
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()

    data_cfg = DataConfig()
    infer_cfg = InferConfig(checkpoint_path=args.checkpoint)

    # Index + select frame
    idx = H5FrameIndex(args.file, dataset_name=data_cfg.dataset_name).build()
    if args.pose_index < 0 or args.pose_index >= len(idx.frames):
        raise ValueError(f"Invalid pose-index {args.pose_index}, file has {len(idx.frames)} frames.")

    fm = idx.frames[args.pose_index]

    # Read only this frame
    fields = ["distance_cm", "azimuth_raw", "elevation_raw", "reflectivity", "r", "g", "b"]
    d = read_frame_fields(args.file, fm.start, fm.end, data_cfg.dataset_name, fields)

    valid = d["distance_cm"] > 0
    if not np.any(valid):
        print("Frame has 0 valid points.")
        return

    dist = d["distance_cm"][valid]
    az = d["azimuth_raw"][valid]
    el = d["elevation_raw"][valid]
    xyz = spherical_to_local_cartesian_np(dist, az, el)  # (N,3) meters

    # Features for model (if needed)
    inten = d["reflectivity"][valid].astype(np.float32) / 255.0
    feats = np.concatenate([xyz, inten[:, None]], axis=1)  # (N,4)

    # Labels: GT or PRED
    if args.mode == "gt":
        pred = rgb_to_class_id(d["r"][valid], d["g"][valid], d["b"][valid]).astype(np.int64)
        point_colors = np.stack([d["r"][valid], d["g"][valid], d["b"][valid]], axis=1).astype(np.float32) / 255.0

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        infer_cfg.device = device.type

        model = PointNetSeg(in_channels=4, num_classes=NUM_CLASSES)
        model = load_model(infer_cfg.checkpoint_path, model, device=device)

        x_full = torch.from_numpy(feats.T).unsqueeze(0).float().to(device)  # (1,4,N)
        N = feats.shape[0]
        Ng = min(data_cfg.num_points_global, N)
        idxg = np.random.choice(N, size=Ng, replace=(N < Ng))
        x_global = x_full[:, :, idxg]

        logits = model.predict_full_cloud(x_full=x_full, x_global=x_global, chunk_size=infer_cfg.batch_points_chunk)
        pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)

        # Points en gris (ou mets une colormap si tu veux)
        point_colors = np.full((len(xyz), 3), 0.7, dtype=np.float32)

    # Build boxes (skip background)
    mask_obs = pred != BACKGROUND_ID
    xyz_obs = xyz[mask_obs]
    pred_obs = pred[mask_obs]
    bboxes = cluster_and_build_bboxes(xyz_obs, pred_obs, infer_cfg.cluster)
    print(f"Frame {args.pose_index}: points={len(xyz)}, obstacles_pts={len(xyz_obs)}, bboxes={len(bboxes)}")

    # Open3D visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float64))

    geoms = [pcd]

    for b in bboxes:
        center = np.array([b.cx, b.cy, b.cz], dtype=np.float32)
        R = yaw_to_R(b.yaw)
        extent = np.array([b.width, b.length, b.height], dtype=np.float32)

        obb = o3d.geometry.OrientedBoundingBox(center=center.astype(np.float64),
                                               R=R.astype(np.float64),
                                               extent=extent.astype(np.float64))
        obb.color = CLASS_ID_TO_RGB.get(b.class_id, np.array([1.0, 1.0, 1.0], dtype=np.float32)).tolist()
        geoms.append(obb)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{args.mode.upper()} Boxes - pose {args.pose_index}", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
