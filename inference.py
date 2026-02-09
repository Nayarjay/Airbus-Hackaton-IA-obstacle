import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN
import lidar_utils

# Imports de nos modules
from model import PointNetSeg
from box_utils import get_oriented_bbox, DBSCAN_PARAMS, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POINTS_INFERENCE = 4096  # Doit correspondre à l'entraînement


def run_inference(h5_file, model_path):
    print(f"--- Inférence sur {h5_file} ---")

    # 1. Charger le modèle
    model = PointNetSeg(num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Charger les données (Sans labels)
    try:
        df = lidar_utils.load_h5_data(h5_file)
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return

    xyz = lidar_utils.spherical_to_local_cartesian(df)
    unique_poses = lidar_utils.get_unique_poses(df)

    all_detections = []

    print(f"Traitement de {len(unique_poses)} frames...")

    with torch.no_grad():
        for idx, pose in unique_poses.iterrows():
            # A. Extraire points de la frame
            mask = (df['ego_x'] == pose['ego_x']) & (df['ego_y'] == pose['ego_y'])
            frame_points = xyz[mask]

            if len(frame_points) == 0: continue

            # B. Préparer pour le modèle (Centrage + Sampling si besoin)
            # Note: Pour l'inférence optimale, on devrait traiter tous les points par morceaux
            # Ici on sample pour aller vite, comme à l'entraînement
            if len(frame_points) >= NUM_POINTS_INFERENCE:
                choice = np.random.choice(len(frame_points), NUM_POINTS_INFERENCE, replace=False)
            else:
                choice = np.random.choice(len(frame_points), NUM_POINTS_INFERENCE, replace=True)

            sampled_points = frame_points[choice]
            centroid = np.mean(sampled_points, axis=0)
            centered_points = sampled_points - centroid

            # Tensor (1, 3, N)
            input_tensor = torch.from_numpy(centered_points).float().transpose(0, 1).unsqueeze(0).to(DEVICE)

            # C. Prédiction
            outputs = model(input_tensor)
            # Argmax pour avoir la classe (0-4)
            preds = outputs.max(1)[1].cpu().numpy()[0]

            # D. Clustering et BBox
            # On cherche les classes 1, 2, 3, 4 (car 0 = background)
            for pred_class_id in [1, 2, 3, 4]:
                # On récupère les points qui ont été classifiés comme tel
                class_mask = (preds == pred_class_id)
                points_of_interest = sampled_points[
                    class_mask]  # Vrais coords (non centrées ici, ou attention au décalage)

                # Attention: sampled_points est décalé ? Non, 'centered_points' l'est.
                # On utilise 'sampled_points' qui sont les coords locales réelles.

                real_class_id = pred_class_id - 1  # Pour revenir à 0-3 (Antenne=0)

                params = DBSCAN_PARAMS[real_class_id]

                if len(points_of_interest) < params['min_samples']:
                    continue

                clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(points_of_interest)
                labels = clustering.labels_

                for label in set(labels):
                    if label == -1: continue

                    cluster_pts = points_of_interest[labels == label]
                    bbox = get_oriented_bbox(cluster_pts)

                    if bbox:
                        # ego_x,y,z,yaw, box_cx,cy,cz, w,l,h, box_yaw, class_id, label
                        row = [
                            pose['ego_x'], pose['ego_y'], pose['ego_z'], pose['ego_yaw'],
                            bbox[0], bbox[1], bbox[2],
                            bbox[3], bbox[4], bbox[5],
                            bbox[6],
                            real_class_id,
                            CLASS_NAMES[real_class_id]
                        ]
                        all_detections.append(row)

            if idx % 10 == 0:
                print(f"Frame {idx} traitée.")

    # 3. Sauvegarde CSV
    columns = [
        'ego_x', 'ego_y', 'ego_z', 'ego_yaw',
        'bbox_center_x', 'bbox_center_y', 'bbox_center_z',
        'bbox_width', 'bbox_length', 'bbox_height', 'bbox_yaw',
        'Class ID', 'Class Label'
    ]

    out_df = pd.DataFrame(all_detections, columns=columns)
    output_filename = h5_file.replace(".h5", "_predictions.csv")
    out_df.to_csv(output_filename, index=False)
    print(f"✅ Terminé ! {output_filename} généré.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Fichier .h5 d'évaluation")
    parser.add_argument("--model", default="pointnet_airbus.pth", help="Chemin du modèle")
    args = parser.parse_args()

    run_inference(args.file, args.model)