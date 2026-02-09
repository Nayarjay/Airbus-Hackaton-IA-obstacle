import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN
import lidar_utils
import tqdm  # Barre de progression

# Vos modules
from model import PointNetSeg
from box_utils import get_oriented_bbox, DBSCAN_PARAMS, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POINTS_INFERENCE = 4096

# Taille des blocs pour le découpage (en mètres)
BLOCK_SIZE_X = 20.0
BLOCK_SIZE_Y = 20.0
STRIDE = 10.0  # Chevauchement (50%) pour ne pas couper un objet en deux


def run_inference_blocked(h5_file, model_path):
    print(f"--- Inférence par blocs sur {h5_file} ---")

    # 1. Charger le modèle
    model = PointNetSeg(num_classes=5).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except:
        print("Erreur: Impossible de charger le modèle. Vérifiez le chemin.")
        return
    model.eval()

    # 2. Charger les données
    try:
        df = lidar_utils.load_h5_data(h5_file)
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return

    xyz = lidar_utils.spherical_to_local_cartesian(df)
    # On ajoute XYZ au dataframe pour faciliter le découpage
    df['x'], df['y'], df['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    unique_poses = lidar_utils.get_unique_poses(df)
    all_detections = []

    print(f"Traitement de {len(unique_poses)} frames...")

    # Boucle sur les frames
    for idx, pose in tqdm.tqdm(unique_poses.iterrows(), total=len(unique_poses)):

        # A. Extraire points de la frame entière
        # Optimisation : On filtre d'abord grossièrement
        frame_mask = (df['ego_x'] == pose['ego_x']) & (df['ego_y'] == pose['ego_y'])
        frame_points = df[frame_mask]

        if len(frame_points) == 0: continue

        frame_xyz = frame_points[['x', 'y', 'z']].values

        # B. Définir la grille de découpage
        min_x, max_x = frame_xyz[:, 0].min(), frame_xyz[:, 0].max()
        min_y, max_y = frame_xyz[:, 1].min(), frame_xyz[:, 1].max()

        # On parcourt la scène par blocs de 20m x 20m
        x_range = np.arange(min_x, max_x, STRIDE)
        y_range = np.arange(min_y, max_y, STRIDE)

        frame_detections = []  # Pour éviter les doublons dans la frame

        with torch.no_grad():
            for x_start in x_range:
                for y_start in y_range:
                    x_end, y_end = x_start + BLOCK_SIZE_X, y_start + BLOCK_SIZE_Y

                    # C. Extraire les points DU BLOC
                    mask_block = (frame_xyz[:, 0] >= x_start) & (frame_xyz[:, 0] < x_end) & \
                                 (frame_xyz[:, 1] >= y_start) & (frame_xyz[:, 1] < y_end)

                    block_points = frame_xyz[mask_block]

                    # Si le bloc est vide ou presque, on passe
                    if len(block_points) < 50: continue

                    # D. Préparer pour le modèle (Sampling + Centrage LOCAL)
                    if len(block_points) >= NUM_POINTS_INFERENCE:
                        choice = np.random.choice(len(block_points), NUM_POINTS_INFERENCE, replace=False)
                    else:
                        choice = np.random.choice(len(block_points), NUM_POINTS_INFERENCE, replace=True)

                    sampled_points = block_points[choice]

                    # IMPORTANT : On centre par rapport au BLOC, pas à la voiture
                    block_center = np.mean(sampled_points, axis=0)
                    centered_points = sampled_points - block_center

                    # Tensor (1, 3, N)
                    input_tensor = torch.from_numpy(centered_points).float().transpose(0, 1).unsqueeze(0).to(DEVICE)

                    # E. Prédiction
                    outputs = model(input_tensor)
                    preds = outputs.max(1)[1].cpu().numpy()[0]

                    # F. Analyse des prédictions du bloc
                    for pred_class_id in [1, 2, 3, 4]:  # On ignore 0 (Fond)

                        class_mask = (preds == pred_class_id)
                        # On récupère les points réels (non centrés)
                        points_of_interest = sampled_points[class_mask]

                        real_class_id = pred_class_id - 1  # 0-3 pour le CSV final
                        params = DBSCAN_PARAMS[real_class_id]

                        if len(points_of_interest) < params['min_samples']: continue

                        # Clustering DBSCAN sur ce bout d'objet
                        clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(
                            points_of_interest)
                        labels = clustering.labels_

                        for label in set(labels):
                            if label == -1: continue

                            cluster_pts = points_of_interest[labels == label]

                            # Calcul BBox
                            bbox = get_oriented_bbox(cluster_pts)

                            if bbox:
                                # On stocke temporairement pour filtrer les doublons plus tard
                                # Format interne simple
                                frame_detections.append({
                                    'bbox': bbox,
                                    'class_id': real_class_id,
                                    'center': np.array([bbox[0], bbox[1], bbox[2]])
                                })

        # G. Fusion des doublons (Non-Maximum Suppression simplifiée)
        # Comme les blocs se chevauchent, on peut détecter le même poteau 2 fois.
        # On va fusionner les boîtes très proches.
        kept_detections = []
        for det in frame_detections:
            is_new = True
            for kept in kept_detections:
                # Si même classe et centres très proches (< 2m)
                dist = np.linalg.norm(det['center'] - kept['center'])
                if det['class_id'] == kept['class_id'] and dist < 2.0:
                    is_new = False
                    break
            if is_new:
                kept_detections.append(det)

        # H. Enregistrement final pour cette frame
        for det in kept_detections:
            bbox = det['bbox']
            cid = det['class_id']
            row = [
                pose['ego_x'], pose['ego_y'], pose['ego_z'], pose['ego_yaw'],
                bbox[0], bbox[1], bbox[2],
                bbox[3], bbox[4], bbox[5],
                bbox[6],
                cid,
                CLASS_NAMES[cid]
            ]
            all_detections.append(row)

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
    print(f"✅ Terminé ! {output_filename} généré avec découpage par blocs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Fichier .h5 d'évaluation")
    parser.add_argument("--model", default="pointnet_airbus.pth", help="Chemin du modèle")
    args = parser.parse_args()

    run_inference_blocked(args.file, args.model)