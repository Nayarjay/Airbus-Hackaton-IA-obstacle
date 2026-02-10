import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# --- CONFIGURATION ---

# Mapping des couleurs (R, G, B) vers ID Classe
COLOR_TO_CLASS_ID = {
    (38, 23, 180): 0,  # Antenna
    (177, 132, 47): 1,  # Cable
    (129, 81, 97): 2,  # Electric pole
    (66, 132, 9): 3  # Wind turbine
}

CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Electric pole", 3: "Wind turbine"}

# Paramètres DBSCAN (Ajustés pour être tolérants)
DBSCAN_PARAMS = {
    0: {'eps': 2.0, 'min_samples': 5},  # Antenna
    1: {'eps': 1.5, 'min_samples': 3},  # Cable (Tolérant pour les lignes fines)
    2: {'eps': 1.5, 'min_samples': 5},  # Pole
    3: {'eps': 3.5, 'min_samples': 10}  # Turbine
}


def get_oriented_bbox(points_xyz):
    """
    Calcule la bounding box orientée.
    Version robuste qui ne plante pas si les points sont colinéaires ou superposés.
    """
    if len(points_xyz) < 3:
        return None

    # 1. Projection 2D (XY)
    points_2d = points_xyz[:, :2]

    # --- PROTECTION ANTI-CRASH PCA ---
    try:
        pca = PCA(n_components=2)
        pca.fit(points_2d)

        # Vérification si la variance est nulle (points tous au même endroit)
        if np.sum(pca.explained_variance_) < 1e-9:
            yaw = 0.0
        else:
            vec = pca.components_[0]
            yaw = np.arctan2(vec[1], vec[0])

    except Exception:
        # En cas d'erreur mathématique quelconque, on met la boîte droite
        yaw = 0.0

    # 2. Rotation des points
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    rotated_xy = points_2d @ R.T

    # 3. Calcul des dimensions
    min_xy = rotated_xy.min(axis=0)
    max_xy = rotated_xy.max(axis=0)
    min_z = points_xyz[:, 2].min()
    max_z = points_xyz[:, 2].max()

    dims_xy = max_xy - min_xy
    height = max_z - min_z

    # Sécurité: éviter les boîtes plates (épaisseur 0)
    dims_xy[0] = max(dims_xy[0], 0.1)
    dims_xy[1] = max(dims_xy[1], 0.1)
    height = max(height, 0.1)

    # 4. Centre
    center_aligned_x = min_xy[0] + dims_xy[0] / 2
    center_aligned_y = min_xy[1] + dims_xy[1] / 2
    center_z = min_z + height / 2

    # Retour au repère monde
    center_world_xy = np.array([center_aligned_x, center_aligned_y]) @ np.linalg.inv(R).T

    return [
        center_world_xy[0], center_world_xy[1], center_z,
        dims_xy[1], dims_xy[0], height,  # W, L, H
        yaw
    ]