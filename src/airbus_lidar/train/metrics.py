from __future__ import annotations
import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: (B,K,N)
    y: (B,N)
    """
    pred = logits.argmax(dim=1)
    correct = (pred == y).float().mean().item()
    return float(correct)
