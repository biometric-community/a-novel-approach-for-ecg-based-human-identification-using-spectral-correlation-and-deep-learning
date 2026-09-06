"""Evaluate a checkpoint (IDR / FAR / FRR / CMC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from ecg_scf.data.datasets import SCFImageDataset, build_scf_arrays, load_subjects
from ecg_scf.engine.loop import evaluate, get_device, load_checkpoint, set_seed
from ecg_scf.models.cnn import build_model
from ecg_scf.utils import load_config, project_root


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ECG SCF CNN checkpoint")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    p.add_argument("--database", type=str, default=None)
    p.add_argument("--fold", type=int, default=0, help="Which CV fold hold-out to score")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.database:
        cfg.setdefault("data", {})["database"] = args.database

    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = get_device()

    subjects = load_subjects(cfg)
    X, y, class_names = build_scf_arrays(subjects, cfg)

    skf = StratifiedKFold(
        n_splits=int(cfg.get("train", {}).get("n_folds", 5)),
        shuffle=True,
        random_state=int(cfg.get("train", {}).get("seed", 42)),
    )
    splits = list(skf.split(X, y))
    fold = int(args.fold) % len(splits)
    _, te = splits[fold]
    test_loader = DataLoader(
        SCFImageDataset(X[te], y[te]),
        batch_size=int(cfg.get("train", {}).get("batch_size", 32)),
        shuffle=False,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root() / ckpt_path
    meta_peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_classes = len(meta_peek.get("meta", {}).get("class_names", class_names))
    model = build_model(num_classes=n_classes, cfg=cfg).to(device)
    meta = load_checkpoint(ckpt_path, model, device)

    metrics = evaluate(
        model,
        test_loader,
        device,
        threshold=float(cfg.get("eval", {}).get("accept_threshold", 0.5)),
        max_rank=int(cfg.get("eval", {}).get("cmc_ranks", 10)),
        far_frr_mode=str(cfg.get("eval", {}).get("far_frr_mode", "eer")),
    )
    out = {
        "checkpoint": str(ckpt_path),
        "fold": fold,
        "metrics": metrics,
        "meta": {k: meta.get(k) for k in ("arch", "database")},
    }
    log_dir = project_root() / cfg.get("paths", {}).get("log_dir", "outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["metrics"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
