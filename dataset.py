import torch
from torch.utils.data import Dataset
import numpy as np
import lidar_utils  # Ton fichier fourni par Airbus
from box_utils import COLOR_TO_CLASS_ID


class AirbusLidarDataset(Dataset):
    def __init__(self, h5_file, num_points=4096, training=True):
        """
        Args:
            h5_file: Chemin vers le fichier .h5
            num_points: Nombre de points fixes à envoyer au réseau (sampling)
            training: Si True, on charge les labels (couleurs). Si False, non.
        """
        self.num_points = num_points
        self.training = training

        print(f"Chargement des données {h5_file}...")
        self.df = lidar_utils.load_h5_data(h5_file)

        # Conversion coordonnées Sphériques -> Cartésiennes
        self.xyz = lidar_utils.spherical_to_local_cartesian(self.df)

        # Préparation des labels si on est en entraînement
        if self.training:
            print("Génération des labels depuis les couleurs...")
            # 0 = Background (tout ce qui n'est pas dans COLOR_TO_CLASS_ID)
            # 1 = Antenna, 2 = Cable, etc. (On décale de +1 pour laisser 0 au fond)
            self.labels = np.zeros(len(self.df), dtype=np.int64)

            for rgb, class_id in COLOR_TO_CLASS_ID.items():
                mask = (self.df['r'] == rgb[0]) & \
                       (self.df['g'] == rgb[1]) & \
                       (self.df['b'] == rgb[2])
                # On met class_id + 1 car 0 est réservé au background
                self.labels[mask] = class_id + 1

        # Identifier les frames (Poses) uniques
        self.poses = lidar_utils.get_unique_poses(self.df)
        self.pose_indices = self.poses['pose_index'].values

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        # 1. Récupérer les données de la frame courante
        pose_row = self.poses.iloc[idx]

        # Filtre rapide (masque)
        mask = (self.df['ego_x'] == pose_row['ego_x']) & \
               (self.df['ego_y'] == pose_row['ego_y'])

        points = self.xyz[mask]

        if self.training:
            targets = self.labels[mask]
        else:
            targets = np.zeros(len(points), dtype=np.int64)  # Dummy targets

        # 2. Sampling (Avoir toujours le même nombre de points, ex: 4096)
        # Si trop de points, on coupe. Si pas assez, on duplique.
        current_n = len(points)
        if current_n >= self.num_points:
            choice = np.random.choice(current_n, self.num_points, replace=False)
        else:
            choice = np.random.choice(current_n, self.num_points, replace=True)

        points = points[choice]
        targets = targets[choice]

        # 3. Normalisation (Centrer les points pour aider le réseau)
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # 4. Conversion Tensor
        # PointNet attend [Channel, N] -> [3, N]
        points_tensor = torch.from_numpy(points).float().transpose(0, 1)
        targets_tensor = torch.from_numpy(targets).long()

        return points_tensor, targets_tensor