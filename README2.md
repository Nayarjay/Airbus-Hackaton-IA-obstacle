# Airbus Hackathon IA — Détection d'obstacles LiDAR

## Groupe : IAobstacle

Membres : Mohamed Rayan Boudiba, Hugo Dubois, Ahmed Sef, Ahmet Emin Cifci, Cheikh Sarr

## Description

Ce projet propose un pipeline complet de détection et classification d'obstacles 3D à partir de nuages de points LiDAR, développé dans le cadre du Hackathon Airbus Helicopters.

Le modèle (PointNet) segmente chaque point du nuage en **5 classes** :
| ID | Classe |
|----|--------|
| 0 | Antenne |
| 1 | Câble |
| 2 | Poteau électrique |
| 3 | Éolienne |
| 4 | Arrière-plan |

Les points classifiés sont ensuite regroupés par clustering DBSCAN pour générer des **bounding boxes 3D orientées (OBB)**, exportées au format CSV.

## Installation

```bash
pip install -r requirements.txt
```

> Pour le support GPU (CUDA 12.1) :
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

## Scripts — Ordre d'exécution

Les scripts se trouvent dans le dossier `scripts/` et s'exécutent dans l'ordre suivant :

### 1. `train_seg.py` — Entraînement du modèle

Entraîne le modèle PointNet sur les données d'entraînement (fichiers HDF5 dans `airbus_hackathon_trainingdata/`). Le script reprend automatiquement l'entraînement à partir du dernier meilleur checkpoint sauvegardé. Les checkpoints sont enregistrés dans `checkpoints/`.

```bash
python scripts/train_seg.py
```

### 2. `visualize_bboxes.py` — Visualisation et analyse des résultats

Permet à l'utilisateur d'inspecter visuellement les résultats frame par frame dans une fenêtre 3D interactive (Open3D). Deux modes disponibles :
- **`gt`** : affiche les labels ground truth (couleurs RGB du dataset)
- **`pred`** : affiche les prédictions du modèle avec les bounding boxes générées

**Contrôles clavier :**
- Flèches gauche/droite ou A/D : frame précédente/suivante
- Flèches haut/bas ou W/S : sauter de ±10 frames

Exemple de paramètres à ajouter dans la configuration : --file "C:\Users\hugom\PycharmProjects\Airbus-Hackaton-IA-obstacle\airbus_hackathon_evalset\eval_sceneA_100.h5" --pose-index 1 --mode pred

```bash
# Visualiser les labels ground truth
python scripts/visualize_bboxes.py --file <chemin_fichier.h5> --pose-index <frame de depart souhaitée> --mode gt

# Visualiser les prédictions du modèle
python scripts/visualize_bboxes.py --file <chemin_fichier.h5> --pose-index <frame de depart souhaitée> --mode pred
```

### 3. `run_eval_export.py` — Inférence et export CSV

Exécute l'inférence sur l'ensemble des fichiers HDF5 d'évaluation (2 scènes × 4 densités de points) et exporte les prédictions sous forme de fichiers CSV dans `predictions_csv/`.

Les fichiers d'évaluation traités (à modifier en fonction de ce sur quoi on souhaite faire l'inférence) :
- `eval_sceneA_100.h5`, `eval_sceneA_75.h5`, `eval_sceneA_50.h5`, `eval_sceneA_25.h5`
- `eval_sceneB_100.h5`, `eval_sceneB_75.h5`, `eval_sceneB_50.h5`, `eval_sceneB_25.h5`

```bash
python scripts/run_eval_export.py
```

## Autres scripts utilitaires

| Script | Description |
|--------|-------------|
| `scripts/build_index.py` | Pré-indexe les fichiers HDF5 pour accélérer le chargement des frames |
| `scripts/infer_export.py` | Inférence sur un fichier unique (exemple : `scene_1.h5`) avec export CSV |
| `visualize.py` (racine) | Visualisation simple d'un nuage de points sans bounding boxes |

## Format de sortie CSV

```
ego_x, ego_y, ego_z, ego_yaw, bbox_center_x, bbox_center_y, bbox_center_z, bbox_width, bbox_length, bbox_height, bbox_yaw, Class ID, Class Label
```