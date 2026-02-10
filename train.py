import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import LidarH5Dataset
from model_pointnet import PointNetSeg
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
    batch_size = 16 
    n_points = 1024
    epochs = 30
    learning_rate = 0.001
    num_classes = 5 # Antenna, Cable, Pole, Turbine, Background
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    data_path = r"airbus_hackathon_trainingdata"
    dataset = LidarH5Dataset(data_path, n_points=n_points, use_cache=True)
    
    if len(dataset) == 0:
        print("No data found!")
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = PointNetSeg(k=num_classes, feature_transform=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Class Weights: [Antenna, Cable, Pole, Turbine, Background]
    # Background (4) is very common, giving it lower weight.
    # Cables (1) are thin and hard, giving highest weight.
    weights = torch.tensor([5.0, 10.0, 5.0, 5.0, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_iou = 0.0
    if not os.path.exists("models"):
        os.makedirs("models")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        for points, target in tqdm(train_loader, desc="Train"):
            points, target = points.to(device), target.to(device)
            # points: (B, 3, N)
            # target: (B, N)
            
            optimizer.zero_grad()
            
            pred, trans_feat = model(points) # (B, N, C)
            
            # Reshape for Loss: (B, C, N) vs (B, N)
            # PointNetSeg returns (B, N, C) typical for softmax last dim, but CrossEntropy expects (B, C, N) or (N, C) vs (N)
            # Let's flatten: (B*N, C) vs (B*N)
            pred = pred.view(-1, num_classes)
            target = target.view(-1)
            
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
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        # IoU tracking
        # We need Intersection and Union for each class
        intersect = np.zeros(num_classes)
        union = np.zeros(num_classes)
        
        with torch.no_grad():
            for points, target in tqdm(val_loader, desc="Val"):
                points, target = points.to(device), target.to(device)
                pred, _ = model(points)
                
                pred = pred.view(-1, num_classes)
                target = target.view(-1)
                
                loss = criterion(pred, target)
                val_loss += loss.item()
                
                pred_choice = pred.data.max(1)[1]
                correct_val += pred_choice.eq(target.data).cpu().sum().item()
                total_val += target.size(0)
                
                # IoU
                p = pred_choice.cpu().numpy()
                t = target.cpu().numpy()
                
                for c in range(num_classes):
                    intersect[c] += np.sum((p == c) & (t == c))
                    union[c] += np.sum((p == c) | (t == c))

        val_acc = correct_val / total_val
        
        # Compute mIoU
        # Filter out classes that were never present in union (avoid div by zero)
        valid_classes = union > 0
        iou_per_class = intersect[valid_classes] / union[valid_classes]
        mIoU = np.mean(iou_per_class)
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | mIoU: {mIoU:.4f}")
        print(f"IoU per class: {iou_per_class}")

        if mIoU >= best_iou:
            best_iou = mIoU
            torch.save(model.state_dict(), 'models/pointnet_segmentation.pth')
            print(f"Saved Best Model (mIoU: {best_iou:.4f})")
            
        scheduler.step()

if __name__ == "__main__":
    train()
