# train.py
import glob
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import NUM_CLASSES, NUM_POINTS
from dataset import LidarFrameDataset
from pointnet_seg import PointNetSeg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_files = sorted(glob.glob("data/train/*.h5"))
    if not train_files:
        raise SystemExit("Aucun fichier trouvé dans data/train/*.h5")

    ds = LidarFrameDataset(train_files, train=True, num_points=NUM_POINTS)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, drop_last=True)

    model = PointNetSeg(num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    os.makedirs("outputs/checkpoints", exist_ok=True)

    model.train()
    for epoch in range(1, 31):
        pbar = tqdm(dl, desc=f"epoch {epoch}")
        loss_sum = 0.0

        for xyz, y, _pose in pbar:
            xyz = xyz.to(device)  # (B,N,3)
            y = y.to(device)      # (B,N)

            logits = model(xyz)   # (B,N,C)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), y.view(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            pbar.set_postfix(loss=float(loss.item()))

        avg = loss_sum / max(1, len(dl))
        print("avg loss:", avg)

        torch.save({"model": model.state_dict()}, f"outputs/checkpoints/epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
