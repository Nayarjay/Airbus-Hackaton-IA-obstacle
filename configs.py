# configs.py

# Mapping classes (adapter si ton README a d'autres IDs/noms)
CLASS_NAMES = {
    0: "Antenna",
    1: "Cable",
    2: "Electric Pole",
    3: "Wind Turbine",
}

# Mapping RGB -> class id (selon ton dataset train)
RGB_TO_CLASS = {
    (38, 23, 180): 0,
    (177, 132, 47): 1,
    (129, 81, 97): 2,
    (66, 132, 9): 3,
}

NUM_CLASSES = 4
NUM_POINTS = 4096

# DBSCAN (post-process)
DBSCAN_EPS = {
    0: 0.8,
    1: 0.6,
    2: 0.9,
    3: 1.2,
}
DBSCAN_MIN_SAMPLES = {
    0: 10,
    1: 10,
    2: 12,
    3: 15,
}
