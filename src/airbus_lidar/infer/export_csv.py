from __future__ import annotations

from typing import List, Dict
import csv

from src.airbus_lidar.geometry.bbox import BBox3D


CSV_COLUMNS = [
    "ego_x", "ego_y", "ego_z", "ego_yaw",
    "bbox_center_x", "bbox_center_y", "bbox_center_z",
    "bbox_width", "bbox_length", "bbox_height",
    "bbox_yaw",
    "Class ID", "Class Label"
]


def write_predictions_csv(out_path: str, rows: List[Dict]) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def bboxes_to_rows(pose: Dict[str, int], bboxes: List[BBox3D]) -> List[Dict]:
    rows = []
    for b in bboxes:
        rows.append({
            "ego_x": pose["ego_x"],
            "ego_y": pose["ego_y"],
            "ego_z": pose["ego_z"],
            "ego_yaw": pose["ego_yaw"],

            "bbox_center_x": b.cx,
            "bbox_center_y": b.cy,
            "bbox_center_z": b.cz,

            "bbox_width": b.width,
            "bbox_length": b.length,
            "bbox_height": b.height,

            "bbox_yaw": b.yaw,

            "Class ID": b.class_id,
            "Class Label": b.class_label,
        })
    return rows
