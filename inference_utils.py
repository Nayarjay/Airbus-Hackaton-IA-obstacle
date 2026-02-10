import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

def get_clusters(points, eps=0.5, min_samples=5):
    """
    Groups points into distinct clusters using DBSCAN.
    Args:
        points: (N, 3) numpy array
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    Returns:
        labels: Cluster labels for each point (-1 means noise)
    """
    if len(points) < min_samples:
        return np.array([])
        
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return db.labels_

def fit_bbox(points):
    """
    Fits an Axis-Aligned Bounding Box (AABB) to a set of points.
    Returns: x, y, z (center), l, w, h (dimensions)
    """
    if len(points) == 0:
        return None
        
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    
    center = (min_pt + max_pt) / 2.0
    dims = max_pt - min_pt
    
    return {
        "center": center,
        "dims": dims, # [length, width, height]
        "min": min_pt,
        "max": max_pt
    }

def get_boxes_from_segmentation(points, labels, num_classes=4):
    """
    Processes segmented points to find individual objects and their boxes.
    Returns a list of boxes: {class_id, center, dims, score}
    """
    all_boxes = []
    
    for class_id in range(num_classes):
        # Filter points belonging to this class
        class_mask = (labels == class_id)
        class_points = points[class_mask]
        
        if len(class_points) < 5:
            continue
            
        # Cluster points to distinguish multiple objects of the same class
        # (e.g. two different poles)
        # Tweak eps based on class (cables need smaller eps, turbines larger)
        eps_map = {0: 1.0, 1: 0.5, 2: 0.8, 3: 1.5} # antenna, cable, pole, turbine
        cluster_labels = get_clusters(class_points, eps=eps_map.get(class_id, 0.5))
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1: continue # Skip noise
            
            cluster_points = class_points[cluster_labels == cluster_id]
            if len(cluster_points) < 5: continue
            
            bbox = fit_bbox(cluster_points)
            if bbox:
                all_boxes.append({
                    "class_id": class_id,
                    "center": bbox["center"],
                    "dims": bbox["dims"],
                    "points": cluster_points
                })
                
    return all_boxes

def create_o3d_box_visual(center, dims, color):
    """Creates an Open3D LineSet representing a box."""
    # center is [x,y,z], dims is [l,w,h]
    # Simple AABB for now, can be upgraded to OOBB
    min_pt = center - dims/2
    max_pt = center + dims/2
    
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_pt, max_pt)
    aabb.color = color
    return aabb
