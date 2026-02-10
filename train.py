import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from pointnet_model import PointNetSegmentation, feature_transform_regularizer, get_model_size
from dataset import get_dataloaders, NUM_CLASSES, CLASS_NAMES


def compute_iou(pred, target, num_classes):
    """Compute IoU per class."""
    ious = []
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    return ious


def train_one_epoch(model, train_loader, criterion, optimizer, device, feature_reg_weight=0.001):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_points = 0

    pbar = tqdm(train_loader, desc="Training")
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, trans3, trans64 = model(features)
        outputs = outputs.reshape(-1, NUM_CLASSES)
        labels = labels.reshape(-1)

        loss = criterion(outputs, labels)

        # Feature transform regularization
        reg_loss = feature_transform_regularizer(trans64)
        loss = loss + feature_reg_weight * reg_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_points += labels.numel()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * total_correct / total_points:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_points

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_points = 0
    all_ious = []

    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(features)
            outputs_flat = outputs.reshape(-1, NUM_CLASSES)
            labels_flat = labels.reshape(-1)

            loss = criterion(outputs_flat, labels_flat)
            total_loss += loss.item()

            pred = outputs_flat.argmax(dim=1)
            total_correct += (pred == labels_flat).sum().item()
            total_points += labels_flat.numel()

            # Compute IoU per batch
            pred_batch = outputs.argmax(dim=2)
            batch_ious = compute_iou(pred_batch, labels, NUM_CLASSES)
            all_ious.append(batch_ious)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_points

    # Average IoU per class
    all_ious = np.array(all_ious)
    mean_ious = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_ious[:4])  # mIoU for obstacle classes only

    return avg_loss, accuracy, mean_ious, miou


def main():
    parser = argparse.ArgumentParser(description="Train PointNet Segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--n_points", type=int, default=32768, help="Number of points per frame")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data loaders
    print("Loading data...")
    train_loader, val_loader, class_weights = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_points=args.n_points,
        num_workers=args.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Class weights: {class_weights}")

    # Model
    model = PointNetSegmentation(num_classes=NUM_CLASSES, input_channels=4)
    model = model.to(device)
    print(f"Model parameters: {get_model_size(model):,}")

    # Loss with class weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0)

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, class_ious, miou = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - start_time

        # Print metrics
        print(f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {100*val_acc:.2f}%, mIoU: {100*miou:.2f}%")
        print("IoU per class:")
        for c in range(NUM_CLASSES):
            iou = class_ious[c]
            if not np.isnan(iou):
                print(f"  {CLASS_NAMES[c]}: {100*iou:.2f}%")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'miou': miou,
            'best_miou': best_miou
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))

        # Save best
        if miou > best_miou:
            best_miou = miou
            checkpoint['best_miou'] = best_miou
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f"New best mIoU: {100*best_miou:.2f}%")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth'))

    print(f"\nTraining complete! Best mIoU: {100*best_miou:.2f}%")


if __name__ == "__main__":
    main()
