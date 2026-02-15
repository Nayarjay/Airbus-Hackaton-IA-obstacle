from __future__ import annotations
import numpy as np


def spherical_to_local_cartesian_np(distance_cm: np.ndarray, azimuth_raw: np.ndarray, elevation_raw: np.ndarray) -> np.ndarray:
    """
    Reprend exactement la convention fournie par Airbus (lidar_utils.spherical_to_local_cartesian)
    distance_cm: cm
    azimuth_raw, elevation_raw: 1/100 deg
    return: (N,3) en mètres
    """
    distance_m = distance_cm.astype(np.float32) / 100.0
    azimuth_rad = np.deg2rad(azimuth_raw.astype(np.float32) / 100.0)
    elevation_rad = np.deg2rad(elevation_raw.astype(np.float32) / 100.0)

    x = distance_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = -distance_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance_m * np.sin(elevation_rad)

    return np.stack([x, y, z], axis=1).astype(np.float32)


def rotate_about_z(points_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return (points_xyz @ R.T).astype(np.float32)
