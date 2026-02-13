import torch
from torch.utils.data import Dataset
import numpy as np
import lidar_utils
from box_utils import COLOR_TO_CLASS_ID


class AirbusLidarDataset(Dataset):
    def __init__(self, h5_file, num_points=4096, training=True):
        self.num_points = num_points
        self.training = training

        # 1. Chargement des données
        # print(f"Chargement des données {h5_file}...")
        # (Commenté pour ne pas spammer la console si on utilise plein de fichiers)
        self.df = lidar_utils.load_h5_data(h5_file)
        self.df = self.df[self.df["distance_cm"] > 0].copy()

        # 2. Conversion en cartésien
        self.xyz = lidar_utils.spherical_to_local_cartesian(self.df)

        # 3. Préparation des labels (1=Antenne, 2=Câble... 0=Fond)
        if self.training:
            self.labels = np.zeros(len(self.df), dtype=np.int64)
            for rgb, class_id in COLOR_TO_CLASS_ID.items():
                mask = (self.df['r'] == rgb[0]) & \
                       (self.df['g'] == rgb[1]) & \
                       (self.df['b'] == rgb[2])
                self.labels[mask] = class_id + 1
        else:
            self.labels = np.zeros(len(self.df), dtype=np.int64)

        # 4. Indexation rapide des frames
        self.poses = lidar_utils.get_unique_poses(self.df)

        # On pré-calcule les indices de chaque frame pour aller vite
        self.frame_indices = []
        # On suppose que le fichier est trié par blocs de frames (sinon groupby est mieux)
        # Pour faire simple et robuste :
        df_group = self.df.groupby(['ego_x', 'ego_y'])
        self.frame_indices = [indices for _, indices in df_group.indices.items()]

    def __len__(self):
        # On multiplie par 10 pour que chaque frame soit vue 10 fois par époque
        # avec des découpages (crops) différents à chaque fois.
        return len(self.frame_indices) * 10

    def __getitem__(self, idx):
        # On récupère la vraie frame (modulo)
        real_idx = idx % len(self.frame_indices)
        indices = self.frame_indices[real_idx]

        points_frame = self.xyz[indices]
        if self.training:
            labels_frame = self.labels[indices]
        else:
            labels_frame = np.zeros(len(points_frame))

        # --- CROP (DÉCOUPAGE) DE 20m x 20m ---
        # Stratégie : On essaie de centrer le crop sur un OBJET (sinon on n'apprend que le sol)

        has_objects = np.any(labels_frame > 0)

        # 80% du temps, si y'a des objets, on se centre dessus
        if self.training and has_objects and np.random.rand() < 0.8:
            obj_indices = np.where(labels_frame > 0)[0]
            center_idx = np.random.choice(obj_indices)
            center_pt = points_frame[center_idx]
        else:
            # Sinon (ou 20% du temps), on se met n'importe où dans la scène
            if len(points_frame) > 0:
                center_idx = np.random.randint(len(points_frame))
                center_pt = points_frame[center_idx]
            else:
                center_pt = np.array([0, 0, 0])

        # On définit la boîte de 20m autour de ce point
        min_x, max_x = center_pt[0] - 10, center_pt[0] + 10
        min_y, max_y = center_pt[1] - 10, center_pt[1] + 10

        mask_crop = (points_frame[:, 0] >= min_x) & (points_frame[:, 0] < max_x) & \
                    (points_frame[:, 1] >= min_y) & (points_frame[:, 1] < max_y)

        crop_points = points_frame[mask_crop]
        crop_labels = labels_frame[mask_crop]

        # Sécurité : Si le crop est vide (bord de map), on prend tout ou on réessaie
        # Ici fallback simple : on prend la frame entière si crop raté
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

        # --- NORMALISATION ---
        # CRUCIAL : On centre les points par rapport à la moyenne DU CROP
        # C'est ce qui permet au modèle de comprendre la forme locale des objets
        centroid = np.mean(final_points, axis=0)
        final_points = final_points - centroid

        # --- DATA AUGMENTATION (Rotation) ---
        # Ça aide le modèle à reconnaître un poteau vu de n'importe quel angle
        if self.training:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            final_points = final_points @ rotation_matrix.T

            # Un peu de bruit (Jitter)
            final_points += np.random.normal(0, 0.01, final_points.shape)

        # Conversion Tensor PyTorch
        points_tensor = torch.from_numpy(final_points).float().transpose(0, 1)
        labels_tensor = torch.from_numpy(final_labels).long()

        return points_tensor, labels_tensor