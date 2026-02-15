from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.airbus_lidar.train.metrics import accuracy


@dataclass
class TrainState:
    best_val_acc: float = 0.0
    epoch: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        amp: bool,
        save_dir: str,
        run_name: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.amp = amp

        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        w = torch.tensor([2.0, 4.0, 2.0, 2.0, 1.0], device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=w)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        self.state = TrainState()

    def _save(self, name: str, **extra: Any) -> None:
        path = self.save_dir / name
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state": self.state.__dict__,
            **extra,
        }
        torch.save(payload, str(path))

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Train {self.state.epoch}", leave=False)
        for batch in pbar:
            x = batch["x"].to(self.device, non_blocking=True)  # (B,C,N)
            y = batch["y"].to(self.device, non_blocking=True)  # (B,N)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                logits = self.model(x)  # (B,K,N)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc = accuracy(logits, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_acc += acc * bs
            n += bs

            pbar.set_postfix(loss=loss.item(), acc=acc)

        return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        for batch in tqdm(self.val_loader, desc="Val", leave=False):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)
            acc = accuracy(logits, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total_acc += acc * bs
            n += bs

        return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}

    def fit(self, epochs: int) -> None:
        for ep in range(epochs):
            self.state.epoch = ep
            tr = self.train_epoch()
            va = self.validate()

            # checkpoint "last"
            self._save(f"{self.run_name}_last.pt", train=tr, val=va)

            # checkpoint "best"
            if va["acc"] > self.state.best_val_acc:
                self.state.best_val_acc = va["acc"]
                self._save(f"{self.run_name}_best.pt", train=tr, val=va)

            print(f"[Epoch {ep}] train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} | val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}")
