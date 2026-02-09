import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

# Imports de nos modules
from dataset import AirbusLidarDataset
from model import PointNetSeg

# --- CONFIG ---
BATCH_SIZE = 8
EPOCHS = 15
LR = 0.001
NUM_POINTS = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(file_path):
    print(f"--- Démarrage de l'entraînement sur {DEVICE} ---")

    # 1. Dataset & DataLoader
    dataset = AirbusLidarDataset(file_path, num_points=NUM_POINTS, training=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Modèle
    # 5 classes = 1 fond + 4 objets
    model = PointNetSeg(num_classes=5).to(DEVICE)

    # 3. Optimiseur & Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Poids pour la Loss : On penalise moins le fond (classe 0) car il est majoritaire
    # On donne plus d'importance aux objets (classes 1-4)
    weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model.train()

    # 4. Boucle d'entraînement
    for epoch in range(EPOCHS):
        total_loss = 0

        for i, (points, targets) in enumerate(dataloader):
            points, targets = points.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            preds = model(points)  # (Batch, 5, N)

            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 5 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS} | Batch {i} | Loss: {loss.item():.4f}")

        print(f"=== Fin Epoch {epoch + 1} | Loss Moyenne: {total_loss / len(dataloader):.4f} ===")

    # 5. Sauvegarde
    save_path = "pointnet_airbus.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé : {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="airbus_hackathon_trainingdata/scene_1.h5", help="Fichier d'entraînement")
    args = parser.parse_args()

    train(args.file)