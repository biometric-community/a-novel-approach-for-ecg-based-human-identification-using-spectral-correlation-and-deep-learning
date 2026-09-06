"""Predict subject ID from a WFDB record or raw .dat/.npy ECG segment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecg_scf.data.datasets import _pick_lead, _resample_to, _zscore, blind_segments
from ecg_scf.engine.loop import get_device, load_checkpoint
from ecg_scf.features.scf import spectral_correlation_image
from ecg_scf.models.cnn import build_model
from ecg_scf.utils import load_config, project_root


def parse_args():
    p = argparse.ArgumentParser(description="Predict identity from ECG via SCF CNN")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    p.add_argument("--input", type=str, required=True, help="WFDB record stem, .npy, or directory")
    p.add_argument("--fs", type=float, default=None, help="Sampling rate if loading raw array")
    return p.parse_args()


def load_signal(path: Path, cfg: dict, fs_override: float | None) -> tuple[np.ndarray, float]:
    data_cfg = cfg.get("data", {})
    target_fs = float(data_cfg.get("target_fs", 360.0))
    prefs = list(data_cfg.get("lead_preference", ["II", "MLII"]))

    if path.suffix.lower() == ".npy":
        sig = np.load(path).astype(np.float64).reshape(-1)
        fs = float(fs_override or target_fs)
        return _zscore(_resample_to(sig, fs, target_fs)), target_fs

    # WFDB stem or .hea
    stem = path.with_suffix("") if path.suffix.lower() == ".hea" else path
    import wfdb

    rec = wfdb.rdrecord(str(stem))
    sig = _pick_lead(list(rec.sig_name or []), rec.p_signal, prefs)
    fs = float(fs_override or rec.fs)
    return _zscore(_resample_to(sig, fs, target_fs)), target_fs


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root() / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("meta", {}).get("class_names")
    if not class_names:
        raise RuntimeError("Checkpoint missing class_names in meta — retrain first")

    model = build_model(num_classes=len(class_names), cfg=cfg).to(device)
    load_checkpoint(ckpt_path, model, device)
    model.eval()

    inp = Path(args.input)
    paths = []
    if inp.is_dir():
        paths = sorted(inp.rglob("*.hea"))
    else:
        paths = [inp]

    img_size = int(cfg.get("model", {}).get("image_size", 128))
    seg_sec = float(cfg.get("data", {}).get("segment_sec", 2.0))
    scf_method = str(cfg.get("scf", {}).get("method", "caf_fft"))
    results = []

    for path in paths:
        sig, fs = load_signal(path, cfg, args.fs)
        segs = blind_segments(sig, fs, seg_sec)
        if not segs:
            results.append({"input": str(path), "error": "no segments"})
            continue
        # Majority vote over segments
        votes = []
        with torch.no_grad():
            for seg in segs[:64]:
                img = spectral_correlation_image(
                    seg, out_size=img_size, method=scf_method
                )
                x = torch.from_numpy(img[None, None, ...]).to(device)
                prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
                votes.append(int(prob.argmax()))
        pred = int(np.bincount(votes).argmax())
        results.append(
            {
                "input": str(path),
                "pred_index": pred,
                "pred_subject": class_names[pred],
                "n_segments": len(segs),
                "voted_segments": len(votes),
            }
        )

    pred_dir = project_root() / cfg.get("paths", {}).get("pred_dir", "outputs/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
