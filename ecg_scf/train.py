"""Train Spectral-Correlation CNN (paper Sec. 5.2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from ecg_scf.data.datasets import SCFImageDataset, build_scf_arrays, load_subjects
from ecg_scf.engine.loop import evaluate, get_device, save_checkpoint, set_seed, train_one_epoch
from ecg_scf.models.cnn import build_model
from ecg_scf.utils import load_config, project_root


def parse_args():
    p = argparse.ArgumentParser(description="Train ECG SCF identification CNN")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--database", type=str, default=None, help="Override data.database")
    p.add_argument("--arch", type=str, default=None, choices=["arch_a", "arch_b"])
    p.add_argument("--max-segments", type=int, default=None, help="Cap segments/subject")
    p.add_argument("--folds", type=int, default=None, help="Override number of CV folds")
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.database:
        cfg.setdefault("data", {})["database"] = args.database
    if args.arch:
        cfg.setdefault("model", {})["name"] = args.arch
    if args.max_segments is not None:
        cfg.setdefault("data", {})["max_segments_per_subject"] = args.max_segments
    if args.folds is not None:
        cfg.setdefault("train", {})["n_folds"] = args.folds
        cfg.setdefault("train", {})["max_folds"] = args.folds
        cfg["train"]["n_validations"] = None  # CLI folds overrides Fig.6 validations
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    train_cfg = cfg.get("train", {})
    set_seed(int(train_cfg.get("seed", 42)))
    device = get_device()

    subjects = load_subjects(cfg)
    print(f"Loaded {len(subjects)} records / subjects from data.database={cfg['data'].get('database')}")
    X, y, class_names = build_scf_arrays(subjects, cfg)
    print(f"SCF images: {X.shape}, classes: {len(class_names)}")

    n_folds = int(train_cfg.get("n_folds", 5))
    max_folds = train_cfg.get("max_folds")
    seed = int(train_cfg.get("seed", 42))
    # Fig. 6: 10 validations; Sec. 5.2: 5-fold 80/20 — use StratifiedShuffleSplit when n_validations set
    n_validations = train_cfg.get("n_validations")
    if n_validations is not None:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(
            n_splits=int(n_validations),
            train_size=float(train_cfg.get("train_ratio", 0.8)),
            random_state=seed,
        )
        splits = list(sss.split(X, y))
        n_folds = len(splits)
    elif n_folds < 2:
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(
            np.arange(len(y)),
            test_size=1.0 - float(train_cfg.get("train_ratio", 0.8)),
            stratify=y,
            random_state=seed,
        )
        splits = [(tr, te)]
        n_folds = 1
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(X, y))

    out_dir = project_root() / cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
    log_dir = project_root() / cfg.get("paths", {}).get("log_dir", "outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    best_idr = -1.0
    best_path = out_dir / "best.pt"

    for fold, (tr, te) in enumerate(splits):
        if max_folds is not None and fold >= int(max_folds):
            break
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        train_ds = SCFImageDataset(X[tr], y[tr])
        test_ds = SCFImageDataset(X[te], y[te])
        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(train_cfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )

        model = build_model(num_classes=len(class_names), cfg=cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(train_cfg.get("lr", 0.002)),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        epochs = int(train_cfg.get("epochs", 15))
        for ep in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  epoch {ep + 1}/{epochs}  loss={loss:.4f}")

        metrics = evaluate(
            model,
            test_loader,
            device,
            threshold=float(cfg.get("eval", {}).get("accept_threshold", 0.5)),
            max_rank=int(cfg.get("eval", {}).get("cmc_ranks", 10)),
            far_frr_mode=str(cfg.get("eval", {}).get("far_frr_mode", "eer")),
        )
        print(
            f"  IDR={metrics['idr']:.4f}  FAR={metrics['far']:.4f}  FRR={metrics['frr']:.4f}"
        )
        fold_metrics.append(metrics)

        fold_path = out_dir / f"fold{fold}.pt"
        meta = {
            "fold": fold,
            "class_names": class_names,
            "metrics": metrics,
            "arch": cfg.get("model", {}).get("name"),
            "database": cfg.get("data", {}).get("database"),
        }
        save_checkpoint(fold_path, model, meta)
        if metrics["idr"] > best_idr:
            best_idr = metrics["idr"]
            save_checkpoint(best_path, model, meta)

    summary = {
        "folds": fold_metrics,
        "mean_idr": float(np.mean([m["idr"] for m in fold_metrics])),
        "mean_far": float(np.mean([m["far"] for m in fold_metrics])),
        "mean_frr": float(np.mean([m["frr"] for m in fold_metrics])),
        "n_classes": len(class_names),
        "database": cfg.get("data", {}).get("database"),
        "arch": cfg.get("model", {}).get("name"),
    }
    summary_path = log_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMean IDR={summary['mean_idr']:.4f}  wrote {summary_path}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
