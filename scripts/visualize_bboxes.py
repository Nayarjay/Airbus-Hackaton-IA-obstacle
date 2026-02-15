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

# Couleurs des boxes = couleurs Airbus (RGB dataset)
CLASS_ID_TO_RGB = {
    cid: np.array(rgb, dtype=np.float32) / 255.0
    for rgb, cid in RGB_TO_CLASS_ID.items()
}

# GLFW keycodes (souvent utilisés par Open3D)
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_UP = 265
KEY_DOWN = 264


def yaw_to_R(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to .h5")
    parser.add_argument("--pose-index", type=int, default=0, help="Frame index (0-based)")
    parser.add_argument("--mode", choices=["gt", "pred"], default="gt", help="gt=labels RGB, pred=model")
    parser.add_argument("--checkpoint", default="checkpoints/pointnet_seg_best.pt", help="Used only if mode=pred")
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()

    data_cfg = DataConfig()
    infer_cfg = InferConfig(checkpoint_path=args.checkpoint)

    # Build frame index once
    idx = H5FrameIndex(args.file, dataset_name=data_cfg.dataset_name).build()
    n_frames = len(idx.frames)
    if n_frames == 0:
        raise RuntimeError("No frames found in file.")
    pose0 = int(np.clip(args.pose_index, 0, n_frames - 1))

    # Load model once (pred mode)
    model = None
    device = None
    if args.mode == "pred":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        infer_cfg.device = device.type
        model = PointNetSeg(in_channels=4, num_classes=NUM_CLASSES)
        model = load_model(infer_cfg.checkpoint_path, model, device=device)

    # State
    state = {
        "pose": pose0,
        "geoms": [],
        "pcd": None,
        "boxes": [],
    }

    def build_geometries(pose_index: int):
        fm = idx.frames[pose_index]

        fields = ["distance_cm", "azimuth_raw", "elevation_raw", "reflectivity", "r", "g", "b"]
        d = read_frame_fields(args.file, fm.start, fm.end, data_cfg.dataset_name, fields)

        valid = d["distance_cm"] > 0
        if not np.any(valid):
            pcd = o3d.geometry.PointCloud()
            return pcd, [], f"pose {pose_index}: 0 valid points"

        dist = d["distance_cm"][valid]
        az = d["azimuth_raw"][valid]
        el = d["elevation_raw"][valid]
        xyz = spherical_to_local_cartesian_np(dist, az, el)  # (N,3)

        # features
        inten = d["reflectivity"][valid].astype(np.float32) / 255.0
        feats = np.concatenate([xyz, inten[:, None]], axis=1)  # (N,4)

        if args.mode == "gt":
            pred = rgb_to_class_id(d["r"][valid], d["g"][valid], d["b"][valid]).astype(np.int64)
            point_colors = np.stack([d["r"][valid], d["g"][valid], d["b"][valid]], axis=1).astype(np.float32) / 255.0
            conf = None
        else:
            # inference full cloud
            x_full = torch.from_numpy(feats.T).unsqueeze(0).float().to(device)  # (1,4,N)
            N = feats.shape[0]
            Ng = min(data_cfg.num_points_global, N)
            idxg = np.random.choice(N, size=Ng, replace=(N < Ng))
            x_global = x_full[:, :, idxg]

            logits = model.predict_full_cloud(x_full=x_full, x_global=x_global, chunk_size=infer_cfg.batch_points_chunk)

            probs = torch.softmax(logits, dim=1)
            conf_t, pred_t = probs.max(dim=1)  # (1,N)
            pred = pred_t.squeeze(0).detach().cpu().numpy().astype(np.int64)
            conf = conf_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # points en gris (tu peux colorer par classe si tu veux)
            point_colors = np.full((len(xyz), 3), 0.70, dtype=np.float32)

        # build boxes
        if conf is None:
            mask_obs = pred != BACKGROUND_ID
        else:
            # seuils par classe (réduit les faux positifs, garde les câbles + permissifs)
            thr = {0: 0.75, 1: 0.55, 2: 0.70, 3: 0.80}
            thr_arr = np.vectorize(lambda c: thr.get(int(c), 0.75))(pred).astype(np.float32)
            mask_obs = (pred != BACKGROUND_ID) & (conf >= thr_arr)

        xyz_obs = xyz[mask_obs]
        pred_obs = pred[mask_obs]
        bboxes = cluster_and_build_bboxes(xyz_obs, pred_obs, infer_cfg.cluster)

        # pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(point_colors.astype(np.float64))

        # boxes
        box_geoms = []
        for b in bboxes:
            center = np.array([b.cx, b.cy, b.cz], dtype=np.float32)
            R = yaw_to_R(b.yaw)
            extent = np.array([b.width, b.length, b.height], dtype=np.float32)

            obb = o3d.geometry.OrientedBoundingBox(center=center.astype(np.float64),
                                                   R=R.astype(np.float64),
                                                   extent=extent.astype(np.float64))
            obb.color = CLASS_ID_TO_RGB.get(b.class_id, np.array([1.0, 1.0, 1.0], dtype=np.float32)).tolist()
            box_geoms.append(obb)

        info = f"pose {pose_index}/{n_frames-1} | points={len(xyz)} | obs_pts={len(xyz_obs)} | bboxes={len(bboxes)} | ego=({fm.ego_x},{fm.ego_y},{fm.ego_z},{fm.ego_yaw})"
        return pcd, box_geoms, info

    def refresh(vis: o3d.visualization.VisualizerWithKeyCallback):
        # keep camera
        vc = vis.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()

        vis.clear_geometries()

        pcd, box_geoms, info = build_geometries(state["pose"])
        state["pcd"] = pcd
        state["boxes"] = box_geoms

        vis.add_geometry(pcd)
        for g in box_geoms:
            vis.add_geometry(g)

        # restore camera (best effort)
        try:
            vc.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
        except TypeError:
            vc.convert_from_pinhole_camera_parameters(cam)

        print(info)
        return False

    def next_frame(vis):
        state["pose"] = min(state["pose"] + 1, n_frames - 1)
        return refresh(vis)

    def prev_frame(vis):
        state["pose"] = max(state["pose"] - 1, 0)
        return refresh(vis)

    def next_10(vis):
        state["pose"] = min(state["pose"] + 10, n_frames - 1)
        return refresh(vis)

    def prev_10(vis):
        state["pose"] = max(state["pose"] - 10, 0)
        return refresh(vis)

    # Open window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"{args.mode.upper()} Boxes - pose {state['pose']}", width=1280, height=720)

    # initial
    refresh(vis)
    opt = vis.get_render_option()
    opt.point_size = float(args.point_size)

    # Key bindings
    vis.register_key_callback(KEY_RIGHT, lambda v: next_frame(v))
    vis.register_key_callback(KEY_LEFT, lambda v: prev_frame(v))
    vis.register_key_callback(ord("D"), lambda v: next_frame(v))
    vis.register_key_callback(ord("A"), lambda v: prev_frame(v))

    # bonus: +/-10 frames
    vis.register_key_callback(KEY_UP, lambda v: next_10(v))
    vis.register_key_callback(KEY_DOWN, lambda v: prev_10(v))
    vis.register_key_callback(ord("W"), lambda v: next_10(v))
    vis.register_key_callback(ord("S"), lambda v: prev_10(v))

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
