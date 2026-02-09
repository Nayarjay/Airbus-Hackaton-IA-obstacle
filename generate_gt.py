import argparse
import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Import de votre librairie fournie par Airbus
import lidar_utils

# --- CONFIGURATION ---
# Couleurs RVB définies dans le README (Section 3)
CLASS_COLORS = {
    0: (38, 23, 180),  # Antenna
    1: (177, 132, 47),  # Cable
    2: (129, 81, 97),  # Electric pole
    3: (66, 132, 9)  # Wind turbine
}

CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Electric pole", 3: "Wind turbine"}

# Paramètres du clustering (A AJUSTER selon vos tests visuels)
DBSCAN_PARAMS = {
    0: {'eps': 1.5, 'min_samples': 5},  # Antenna
    1: {'eps': 1.0, 'min_samples': 3},  # Cable (points très dispersés)
    2: {'eps': 1.2, 'min_samples': 10},  # Pole
    3: {'eps': 3.0, 'min_samples': 15}  # Wind turbine
}


def get_oriented_bbox(points_xyz):
    """
    Calcule la bounding box orientée (x, y, z, w, l, h, yaw) pour un cluster de points.
    """
    if len(points_xyz) < 3:
        return None

    # 1. Projection 2D (XY) pour trouver l'orientation (Yaw) via PCA
    points_2d = points_xyz[:, :2]
    pca = PCA(n_components=2)
    pca.fit(points_2d)

    # Le vecteur propre principal donne l'angle
    vec = pca.components_[0]
    yaw = np.arctan2(vec[1], vec[0])

    # 2. Rotation des points pour les aligner sur les axes X/Y
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    rotated_xy = points_2d @ R.T

    # 3. Calcul des dimensions (min/max)
    min_xy = rotated_xy.min(axis=0)
    max_xy = rotated_xy.max(axis=0)
    min_z = points_xyz[:, 2].min()
    max_z = points_xyz[:, 2].max()

    dims_xy = max_xy - min_xy
    height = max_z - min_z

    # 4. Calcul du centre de la boîte (dans le monde réel)
    center_aligned_x = min_xy[0] + dims_xy[0] / 2
    center_aligned_y = min_xy[1] + dims_xy[1] / 2
    center_z = min_z + height / 2

    # On dé-rotationne le centre XY
    center_world_xy = np.array([center_aligned_x, center_aligned_y]) @ np.linalg.inv(R).T

    return [
        center_world_xy[0], center_world_xy[1], center_z,  # Center X, Y, Z
        dims_xy[1], dims_xy[0], height,  # Width, Length, Height (ordre ajusté)
        yaw  # Yaw
    ]


def main():
    parser = argparse.ArgumentParser(description="Générer les Bounding Boxes (Vérité Terrain) depuis les couleurs RGB")
    parser.add_argument("--file", required=True, help="Chemin vers le fichier .h5")
    args = parser.parse_args()

    print(f"Chargement de {args.file}...")
    try:
        df = lidar_utils.load_h5_data(args.file)
    except Exception as e:
        print(f"Erreur: {e}")
        return

    # 1. Conversion coordonnées Sphériques -> Cartésiennes (via lidar_utils)
    xyz = lidar_utils.spherical_to_local_cartesian(df)

    # On attache les coordonnées XYZ au DataFrame pour filtrer facilement
    df['x'] = xyz[:, 0]
    df['y'] = xyz[:, 1]
    df['z'] = xyz[:, 2]

    # 2. Récupérer la liste des frames (Poses)
    pose_counts = lidar_utils.get_unique_poses(df)
    print(f"Traitement de {len(pose_counts)} frames...")

    all_bboxes = []

    # 3. Boucle sur chaque Frame (Pose)
    for idx, pose_row in pose_counts.iterrows():
        pose_idx = int(pose_row['pose_index'])

        # Filtre les points pour CETTE frame spécifique
        frame_df = lidar_utils.filter_by_pose(df, pose_row)

        if len(frame_df) == 0: continue

        # Pour chaque classe (Antenne, Cable, etc.)
        for class_id, color_rgb in CLASS_COLORS.items():

            # Filtre par couleur (Labels RGB fournis)
            mask_color = (frame_df['r'] == color_rgb[0]) & \
                         (frame_df['g'] == color_rgb[1]) & \
                         (frame_df['b'] == color_rgb[2])

            points_class = frame_df[mask_color]

            if len(points_class) < 5:
                continue  # Pas assez de points pour faire un objet

            # Récupérer XYZ pour le clustering
            X_cluster = points_class[['x', 'y', 'z']].values

            # Clustering DBSCAN
            params = DBSCAN_PARAMS[class_id]
            clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(X_cluster)

            labels = clustering.labels_
            unique_labels = set(labels)

            for label in unique_labels:
                if label == -1: continue  # C'est du bruit, on ignore

                # Points appartenant à cet objet unique
                obj_points = X_cluster[labels == label]

                # Calculer la boîte
                bbox = get_oriented_bbox(obj_points)

                if bbox:
                    # Format de sortie CSV demandé par le README
                    # ego_x, ego_y, ego_z, ego_yaw identification de la frame
                    bbox_row = [
                        pose_row['ego_x'], pose_row['ego_y'], pose_row['ego_z'], pose_row['ego_yaw'],
                        bbox[0], bbox[1], bbox[2],  # Center
                        bbox[3], bbox[4], bbox[5],  # W, L, H
                        bbox[6],  # Yaw
                        class_id, CLASS_NAMES[class_id]
                    ]
                    all_bboxes.append(bbox_row)

        if idx % 10 == 0:
            print(f"Frame {idx}/{len(pose_counts)} traitée...")

    # 4. Sauvegarde en CSV
    columns = [
        'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
        'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
        'bbox_width', 'bbox_length', 'bbox_height',
        'bbox_yaw',
        'Class ID', 'Class Label'
    ]

    out_df = pd.DataFrame(all_bboxes, columns=columns)
    output_filename = args.file.replace(".h5", "_groundtruth.csv")
    out_df.to_csv(output_filename, index=False)

    print(f"\n✅ Terminé ! {len(out_df)} boîtes générées.")
    print(f"Fichier sauvegardé : {output_filename}")


if __name__ == "__main__":
    main()