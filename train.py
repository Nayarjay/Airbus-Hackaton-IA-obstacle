import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import LidarH5Dataset
from model_pointnet import PointNetCls
import numpy as np
from tqdm import tqdm

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def train():
    # Parameters
    batch_size = 32
    n_points = 1024
    epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = LidarH5Dataset(r"airbus_hackathon_trainingdata", n_points=n_points)
    if len(dataset) == 0:
        print("No data found!")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = PointNetCls(k=2, feature_transform=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Check for imbalance and use weights if necessary
    # (Simple approach: count labels in a subset or assume DANGER is rarer)
    criterion = nn.NLLLoss()

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (points, target) in enumerate(tqdm(train_loader)):
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            
            pred, trans_feat = model(points)
            loss = criterion(pred, target)
            if trans_feat is not None:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct += pred_choice.eq(target.data).cpu().sum().item()
            total += target.size(0)

        train_acc = correct / total
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for points, target in val_loader:

                points, target = points.to(device), target.to(device)
                pred, _ = model(points)
                pred_choice = pred.data.max(1)[1]
                val_correct += pred_choice.eq(target.data).cpu().sum().item()
                val_total += target.size(0)
        
        val_acc = val_correct / val_total
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/pointnet_obstacle_avoidance.pth')
            print("Model saved.")

        scheduler.step()

if __name__ == "__main__":
    train()
