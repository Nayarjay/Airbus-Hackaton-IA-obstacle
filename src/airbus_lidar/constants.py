from __future__ import annotations

# 4 classes demandées
CLASS_ID_TO_LABEL = {
    0: "Antenna",
    1: "Cable",
    2: "Electric Pole",
    3: "Wind Turbine",
}

# On ajoute un background pour l'entraînement segmentation
BACKGROUND_ID = 4
NUM_CLASSES = 5  # 0..3 obstacles + 4 background

# Mapping RGB (dataset) -> class_id
RGB_TO_CLASS_ID = {
    (38, 23, 180): 0,    # Antenna
    (177, 132, 47): 1,   # Cable
    (129, 81, 97): 2,    # Electric pole
    (66, 132, 9): 3,     # Wind turbine
}
