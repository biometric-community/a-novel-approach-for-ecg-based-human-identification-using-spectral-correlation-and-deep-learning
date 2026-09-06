"""Training / evaluation loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ecg_scf.metrics.id_metrics import summarize_metrics


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs)
        ys.append(y.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(ys, axis=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = 0.5,
    max_rank: int = 10,
    far_frr_mode: str = "eer",
) -> dict[str, Any]:
    probs, y_true = predict_proba(model, loader, device)
    return summarize_metrics(
        y_true,
        probs,
        threshold=threshold,
        max_rank=max_rank,
        far_frr_mode=far_frr_mode,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt.get("meta", {})
