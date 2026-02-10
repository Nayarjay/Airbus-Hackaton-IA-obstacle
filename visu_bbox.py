import argparse
import numpy as np
import open3d as o3d
import pandas as pd
import lidar_utils

# --- CONFIGURATION DES COULEURS (Format R, V, B entre 0.0 et 1.0) ---
# Correspondance exacte avec le README
CLASS_COLORS = {
    0: [0.0, 0.0, 1.0],  # Antenna (Bleu pur)
    1: [1.0, 0.55, 0.0],  # Cable (Orange vif)
    2: [0.8, 0.0, 0.8],  # Electric pole (Magenta/Violet)
    3: [0.0, 1.0, 0.0]  # Wind turbine (Vert fluo)
}


def create_o3d_bbox(row):
    """Crée une boîte Open3D colorée selon sa classe"""
    # 1. Géométrie (Centre, Taille)
    center = np.array([row['bbox_center_x'], row['bbox_center_y'], row['bbox_center_z']])
    extent = np.array([row['bbox_length'], row['bbox_width'], row['bbox_height']])

    # 2. Orientation (Yaw) -> Matrice de Rotation R
    yaw = row['bbox_yaw']
    c, s = np.cos(yaw), np.sin(yaw)
    # Rotation autour de Z uniquement
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    # 3. Création de l'objet
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)

    # 4. COLORISATION
    # On récupère l'ID (ex: 1.0 -> 1)
    class_id = int(row['Class ID'])

    # On cherche la couleur, sinon Rouge par défaut
    color = CLASS_COLORS.get(class_id, [1.0, 0.0, 0.0])
    obb.color = color  # Open3D attend une liste [r, g, b] floats

    return obb


def main():
    parser = argparse.ArgumentParser(description="Visualisation BBox Colorées")
    parser.add_argument("--file", required=True, help="Fichier .h5 (Points)")
    parser.add_argument("--csv", required=True, help="Fichier .csv (Boîtes)")
    parser.add_argument("--pose-index", type=int, default=0, help="Index de la frame")
    args = parser.parse_args()

    # 1. Chargement
    print(f"--- Chargement ---")
    df_points = lidar_utils.load_h5_data(args.file)
    df_boxes = pd.read_csv(args.csv)

    # 2. Sélection Frame
    poses = lidar_utils.get_unique_poses(df_points)
    if args.pose_index >= len(poses):
        print(f"Index {args.pose_index} invalide.")
        return

    pose_row = poses.iloc[args.pose_index]
    print(f"Frame {args.pose_index} (X={pose_row['ego_x']:.1f}, Y={pose_row['ego_y']:.1f})")

    # 3. Filtrage
    # Points
    frame_points = lidar_utils.filter_by_pose(df_points, pose_row)
    xyz = lidar_utils.spherical_to_local_cartesian(frame_points)

    # Boîtes (Tolérance pour matching flottant)
    # On regarde si le ego_x/y de la boîte est proche de celui de la frame
    dist_x = np.abs(df_boxes['ego_x'] - pose_row['ego_x'])
    dist_y = np.abs(df_boxes['ego_y'] - pose_row['ego_y'])
    frame_boxes = df_boxes[(dist_x < 0.1) & (dist_y < 0.1)]  # 10cm de tolérance

    print(f"Points: {len(xyz)}")
    print(f"Boîtes: {len(frame_boxes)}")

    # 4. Construction de la Scène
    geometries = []

    # A. Nuage de points (Gris pour contraste)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])  # Gris foncé

    # Optionnel: Si on veut utiliser l'intensité réelle
    if "reflectivity" in frame_points.columns:
        intens = frame_points["reflectivity"].to_numpy() / 255.0
        # On met en niveaux de gris
        colors = np.stack([intens, intens, intens], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries.append(pcd)

    # B. Ajout des Boîtes Colorées
    for _, row in frame_boxes.iterrows():
        bbox = create_o3d_bbox(row)
        geometries.append(bbox)

    # C. Repère (Optionnel)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries.append(axes)

    # 5. Rendu Visuel Avancé
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {args.pose_index}", width=1280, height=720)

    for g in geometries:
        vis.add_geometry(g)

    # Options de Rendu
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # FOND NOIR
    opt.point_size = 2.0
    opt.line_width = 5.0  # TRAITS ÉPAIS (Pour bien voir les boîtes)
    opt.show_coordinate_frame = True

    # Vue Caméra (Vue de dessus/arrière)
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([-1.0, 0.0, 1.0])  # Vue oblique
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.1)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()