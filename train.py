import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import glob
import os
import numpy as np
from tqdm import tqdm  # Barre de progression (pip install tqdm)

# Imports de vos modules locaux
from dataset import AirbusLidarDataset
from model import PointNetSeg

# --- CONFIGURATION ---
DATA_DIR = "airbus_hackathon_trainingdata"  # Dossier contenant les .h5
BATCH_SIZE = 16  # Taille du paquet (réduire à 8 si erreur de mémoire GPU)
EPOCHS = 10  # Nombre de tours complets (30-50 est bien)
LR = 0.001  # Vitesse d'apprentissage
NUM_POINTS = 4096  # Doit correspondre à inference.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"--- Démarrage de l'entraînement sur {DEVICE} ---")

    # 1. Lister tous les fichiers .h5
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "scene_*.h5")))

    if len(all_files) == 0:
        print(f"ERREUR: Aucun fichier trouvé dans {DATA_DIR}")
        return

    # 2. Séparation Train / Validation
    # On garde les 2 derniers fichiers pour la validation (ex: scene_9, scene_10)
    # Si on a peu de fichiers, on en garde juste 1
    split_index = max(1, len(all_files) - 2)

    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    print(f"Fichiers d'entraînement ({len(train_files)}) : {[os.path.basename(f) for f in train_files]}")
    print(f"Fichiers de validation ({len(val_files)})   : {[os.path.basename(f) for f in val_files]}")

    # 3. Création des Datasets
    print("Chargement des données... (Cela peut prendre un moment)")

    # On crée une liste de datasets pour chaque fichier
    train_datasets = [AirbusLidarDataset(f, num_points=NUM_POINTS, training=True) for f in train_files]
    val_datasets = [AirbusLidarDataset(f, num_points=NUM_POINTS, training=True) for f in val_files]

    # On les fusionne en un seul gros dataset virtuel
    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset = ConcatDataset(val_datasets)

    # Création des DataLoaders (Le chargeur qui envoie les données par paquets au GPU)
    # num_workers=0 pour compatibilité Windows (mettre 4 sous Linux pour aller plus vite)
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(full_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Initialisation du Modèle
    # 5 classes en sortie : 0=Fond, 1=Antenne, 2=Câble, 3=Poteau, 4=Éolienne
    model = PointNetSeg(num_classes=5).to(DEVICE)

    # Optimiseur (Adam est le standard)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Scheduler : Réduit le Learning Rate quand ça stagne (optionnel mais recommandé)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 5. Gestion du déséquilibre des classes (Weights)
    # C'est ICI qu'on force le modèle à apprendre les câbles
    # Poids : [Fond, Antenne, Câble, Poteau, Éolienne]
    # On met 0.1 au fond pour l'ignorer un peu, et 20.0 aux câbles pour les prioriser
    weights = torch.tensor([0.1, 5.0, 20.0, 10.0, 5.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Variables pour sauvegarder le meilleur modèle
    best_val_loss = float('inf')
    save_path = "pointnet_airbus_best.pth"

    # 6. Boucle d'entraînement
    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch + 1}/{EPOCHS} ===")

        # --- PHASE TRAIN ---
        model.train()
        train_loss = 0.0

        # Barre de progression pour le train
        pbar = tqdm(train_loader, desc="Training")

        for points, targets in pbar:
            points, targets = points.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            preds = model(points)  # (Batch, 5, N)

            # Loss calculation
            loss = criterion(preds, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()  # Mise à jour du learning rate

        # --- PHASE VALIDATION ---
        model.eval()
        val_loss = 0.0
        correct_points = 0
        total_points = 0

        with torch.no_grad():
            for points, targets in val_loader:
                points, targets = points.to(DEVICE), targets.to(DEVICE)

                preds = model(points)
                loss = criterion(preds, targets)
                val_loss += loss.item()

                # Calcul précision (Accuracy) simple
                pred_choice = preds.max(1)[1]  # Argmax
                correct = pred_choice.eq(targets).sum().item()
                total = targets.numel()
                correct_points += correct
                total_points += total

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct_points / total_points

        print(
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy (Global): {accuracy:.2f}%")

        # --- SAUVEGARDE DU MEILLEUR MODÈLE ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Nouveau record ! Modèle sauvegardé sous {save_path}")
        else:
            print(f"Pas d'amélioration (Best: {best_val_loss:.4f})")

    print("\n🎉 Entraînement terminé !")


if __name__ == "__main__":
    # Vérification des dossiers
    if not os.path.exists(DATA_DIR):
        print(f"Attention: Le dossier '{DATA_DIR}' n'existe pas.")
        print("Veuillez vérifier le chemin ou créer le dossier.")
    else:
        train()