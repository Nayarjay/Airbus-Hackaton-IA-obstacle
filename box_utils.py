import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# --- CONFIGURATION ---

# Mapping des couleurs (R, G, B) vers ID Classe (0-3 selon le README)
# Utilisé pour créer la vérité terrain
COLOR_TO_CLASS_ID = {
    (38, 23, 180): 0,  # Antenna
    (177, 132, 47): 1,  # Cable
    (129, 81, 97): 2,  # Electric pole
    (66, 132, 9): 3  # Wind turbine
}

# Noms pour le CSV final
CLASS_NAMES = {0: "Antenna", 1: "Cable", 2: "Electric pole", 3: "Wind turbine"}

# Paramètres DBSCAN par classe (A AJUSTER SELON TES TESTS)
# EPS: Rayon de recherche (mètres)
# MIN_SAMPLES: Nombre min de points pour faire un objet
DBSCAN_PARAMS = {
    0: {'eps': 1.5, 'min_samples': 5},  # Antenna
    1: {'eps': 1.0, 'min_samples': 3},  # Cable
    2: {'eps': 1.2, 'min_samples': 10},  # Pole
    3: {'eps': 3.0, 'min_samples': 15}  # Turbine
}


def get_oriented_bbox(points_xyz):
    """
    Calcule la bounding box orientée (x, y, z, w, l, h, yaw) pour un nuage de points.
    Retourne None si pas assez de points.
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
        dims_xy[1], dims_xy[0], height,  # Width, Length, Height
        yaw  # Yaw
    ]