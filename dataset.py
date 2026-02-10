import torch
from torch.utils.data import Dataset
import numpy as np
import lidar_utils
from box_utils import COLOR_TO_CLASS_ID


class AirbusLidarDataset(Dataset):
    def __init__(self, h5_file, num_points=4096, training=True):
        self.num_points = num_points
        self.training = training

        print(f"Chargement des données {h5_file}...")
        self.df = lidar_utils.load_h5_data(h5_file)

        # Conversion coordonnées
        self.xyz = lidar_utils.spherical_to_local_cartesian(self.df)

        # Préparation des labels
        if self.training:
            self.labels = np.zeros(len(self.df), dtype=np.int64)
            for rgb, class_id in COLOR_TO_CLASS_ID.items():
                mask = (self.df['r'] == rgb[0]) & \
                       (self.df['g'] == rgb[1]) & \
                       (self.df['b'] == rgb[2])
                self.labels[mask] = class_id + 1  # 0=Fond, 1..4=Objets

        # Frames uniques
        self.poses = lidar_utils.get_unique_poses(self.df)

        # Optimisation : on pré-calcule les masques par frame pour aller vite
        self.frame_indices = []
        for _, pose in self.poses.iterrows():
            mask = (self.df['ego_x'] == pose['ego_x']) & (self.df['ego_y'] == pose['ego_y'])
            self.frame_indices.append(np.where(mask)[0])

    def __len__(self):
        # On peut artificiellement augmenter la taille si on veut plus d'exemples par époque
        return len(self.poses) * 4  # 4 blocs par frame par époque

    def __getitem__(self, idx):
        # On récupère la frame correspondante (modulo le nombre de frames)
        real_idx = idx % len(self.poses)
        indices = self.frame_indices[real_idx]

        points_frame = self.xyz[indices]
        labels_frame = self.labels[indices] if self.training else np.zeros(len(points_frame))

        # --- STRATÉGIE DE DÉCOUPAGE (CROP) ---
        # On veut un bloc de 20m x 20m (comme dans inference.py)

        # Pour aider le modèle, on force le bloc à être centré sur un OBJET
        # 70% du temps (sinon il ne verra que du sol et apprendra rien)
        has_objects = np.any(labels_frame > 0)

        if self.training and has_objects and np.random.rand() < 0.7:
            # On centre sur un objet au hasard
            obj_indices = np.where(labels_frame > 0)[0]
            center_idx = np.random.choice(obj_indices)
            center_pt = points_frame[center_idx]
        else:
            # On centre n'importe où (random)
            if len(points_frame) > 0:
                center_idx = np.random.randint(len(points_frame))
                center_pt = points_frame[center_idx]
            else:
                center_pt = np.array([0, 0, 0])

        # Découpage du bloc 20x20m (x +/- 10m, y +/- 10m)
        min_x, max_x = center_pt[0] - 10, center_pt[0] + 10
        min_y, max_y = center_pt[1] - 10, center_pt[1] + 10

        mask_crop = (points_frame[:, 0] >= min_x) & (points_frame[:, 0] < max_x) & \
                    (points_frame[:, 1] >= min_y) & (points_frame[:, 1] < max_y)

        crop_points = points_frame[mask_crop]
        crop_labels = labels_frame[mask_crop]

        # Si le bloc est vide ou trop petit, on prend tout (fallback)
        if len(crop_points) < 50:
            crop_points = points_frame
            crop_labels = labels_frame

        # --- SAMPLING (4096 points) ---
        if len(crop_points) >= self.num_points:
            choice = np.random.choice(len(crop_points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(crop_points), self.num_points, replace=True)

        final_points = crop_points[choice]
        final_labels = crop_labels[choice]

        # --- NORMALISATION (Même logique que inference.py) ---
        # On centre par rapport à la moyenne DU BLOC
        centroid = np.mean(final_points, axis=0)
        final_points = final_points - centroid

        # Data Augmentation (Rotation aléatoire autour de Z)
        if self.training:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            final_points = final_points @ rotation_matrix.T
            # Bruit
            final_points += np.random.normal(0, 0.01, final_points.shape)

        # Tensor
        points_tensor = torch.from_numpy(final_points).float().transpose(0, 1)
        labels_tensor = torch.from_numpy(final_labels).long()

        return points_tensor, labels_tensor