"""PhysioNet ECG loaders + blind 2 s segmentation (Sec. 4.2, Table 4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from ecg_scf.features.scf import spectral_correlation_image
from ecg_scf.utils import project_root, resolve_path


# Catalog id → PhysioNet / local layout hints
DB_ALIASES = {
    "fantasia": "fantasia",
    "stdb": "stdb",
    "vfdb": "vfdb",
    "nsrdb": "nsrdb",
    "mitdb": "mit-bih",
    "mit-bih": "mit-bih",
    "ptbdb": "ptb",
    "ptb": "ptb",
    "cebsdb": "cebsdb",
    "afdb": "afdb",
}


@dataclass
class SubjectRecord:
    subject_id: str
    database: str
    signal: np.ndarray  # 1-D float64 at target_fs
    fs: float


def _datasets_root(cfg: dict) -> Path:
    rel = cfg.get("data", {}).get("datasets_root", "../../datasets")
    return resolve_path(rel, project_root())


def _find_hea_files(db_dir: Path) -> list[Path]:
    if not db_dir.exists():
        return []
    heas = sorted(db_dir.rglob("*.hea"))
    out: list[Path] = []
    for h in heas:
        stem = h.with_suffix("")
        # Skip MIT-BIH "x_*" auxiliary records
        if h.stem.startswith("x_"):
            continue
        if any(stem.with_suffix(s).exists() for s in (".dat", ".ecg", ".ECG")):
            out.append(h)
    # Prefer official RECORDS list when present (e.g. MITDB 48 records)
    records_file = db_dir / "RECORDS"
    if not records_file.exists():
        # sometimes one level down
        cand = list(db_dir.glob("**/RECORDS"))
        records_file = cand[0] if cand else records_file
    if records_file.exists():
        allowed = {ln.strip() for ln in records_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        filtered = [h for h in out if h.stem in allowed]
        if filtered:
            return filtered
    return out

def _pick_lead(sig_names: list[str], p_signal: np.ndarray, prefs: list[str]) -> np.ndarray:
    names = [str(n).strip() for n in (sig_names or [])]
    lower = [n.lower() for n in names]
    for pref in prefs:
        p = pref.lower()
        for i, n in enumerate(lower):
            if n == p or p in n:
                return np.asarray(p_signal[:, i], dtype=np.float64)
    # MIT-BIH often MLII as first channel
    return np.asarray(p_signal[:, 0], dtype=np.float64)


def _resample_to(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample with polyphase FIR (antialiasing) — Sec. 4.2."""
    if abs(fs_in - fs_out) < 1e-6:
        return sig.astype(np.float64)
    from math import gcd

    up = int(round(fs_out))
    down = int(round(fs_in))
    g = gcd(up, down)
    up //= g
    down //= g
    return resample_poly(sig, up, down).astype(np.float64)


def _zscore(sig: np.ndarray) -> np.ndarray:
    """Normalize to remove device / electrode dependency (Sec. 4.2)."""
    mu = float(np.mean(sig))
    sd = float(np.std(sig)) + 1e-8
    return (sig - mu) / sd


def _read_wfdb_record(stem: Path, lead_prefs: list[str]) -> tuple[np.ndarray, float] | None:
    try:
        import wfdb
    except ImportError as e:
        raise ImportError("wfdb is required to load PhysioNet records") from e
    try:
        rec = wfdb.rdrecord(str(stem))
    except Exception:
        return None
    if rec.p_signal is None:
        return None
    fs = float(rec.fs)
    sig = _pick_lead(list(rec.sig_name or []), rec.p_signal, lead_prefs)
    return sig, fs


# MIT-BIH: records 201 and 202 are the same patient → 47 subjects (paper Table 4 / Sec. 5.1)
MITDB_SUBJECT_MAP = {
    "202": "201",
}


def _subject_id_for(catalog: str, record_stem: str, hea: Path) -> str:
    if catalog == "ptb":
        return f"{catalog}:{hea.parent.name}"
    if catalog == "mit-bih":
        sid = MITDB_SUBJECT_MAP.get(record_stem, record_stem)
        return f"{catalog}:{sid}"
    return f"{catalog}:{record_stem}"


def iter_database_records(db_id: str, cfg: dict) -> Iterator[SubjectRecord]:
    """Yield SubjectRecords; multiple WFDB files may map to one subject (PTB, MITDB)."""
    data_cfg = cfg.get("data", {})
    root = _datasets_root(cfg)
    catalog = DB_ALIASES.get(db_id.lower(), db_id.lower())
    db_dir = root / catalog
    lead_prefs = list(data_cfg.get("lead_preference", ["II", "MLII"]))
    target_fs = float(data_cfg.get("target_fs", 360.0))
    max_minutes = float(data_cfg.get("max_minutes", 30.0))
    max_samples = int(target_fs * max_minutes * 60.0)

    heas = _find_hea_files(db_dir)
    if not heas:
        return

    # Accumulate by subject then concatenate (MITDB 201/202, multi-record PTB)
    buckets: dict[str, list[np.ndarray]] = {}
    for hea in heas:
        stem = hea.with_suffix("")
        got = _read_wfdb_record(stem, lead_prefs)
        if got is None:
            continue
        sig, fs = got
        sig = _resample_to(sig, fs, target_fs)
        if sig.size > max_samples:
            sig = sig[:max_samples]
        sig = _zscore(sig)
        sid = _subject_id_for(catalog, stem.name, hea)
        buckets.setdefault(sid, []).append(sig)

    for sid, parts in buckets.items():
        # Cap total length after merge
        sig = np.concatenate(parts)
        if sig.size > max_samples:
            sig = sig[:max_samples]
        yield SubjectRecord(
            subject_id=sid,
            database=catalog,
            signal=sig,
            fs=target_fs,
        )


def blind_segments(sig: np.ndarray, fs: float, segment_sec: float = 2.0) -> list[np.ndarray]:
    """Non-overlapping blind 2 s windows (Sec. 4.2) — no fiducials."""
    win = int(round(fs * segment_sec))
    if win < 8 or sig.size < win:
        return []
    n = (sig.size // win) * win
    segs = []
    for i in range(0, n, win):
        segs.append(sig[i : i + win].astype(np.float64))
    return segs


def synthetic_subjects(n_subjects: int = 8, fs: float = 360.0, minutes: float = 1.0, seed: int = 0) -> list[SubjectRecord]:
    """Demo ECG-like periodic pulses when PhysioNet data is missing."""
    rng = np.random.default_rng(seed)
    out: list[SubjectRecord] = []
    n = int(fs * minutes * 60)
    t = np.arange(n) / fs
    for i in range(n_subjects):
        hr = 60.0 + rng.uniform(-5, 5) + i * 0.3
        f0 = hr / 60.0
        # Subject-specific harmonic mix
        sig = (
            1.0 * np.sin(2 * np.pi * f0 * t)
            + 0.35 * np.sin(2 * np.pi * 2 * f0 * t + i)
            + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
            + 0.02 * rng.standard_normal(n)
        )
        # QRS-like spikes
        period = int(fs / f0)
        for k in range(0, n, max(period, 1)):
            if k + 3 < n:
                sig[k : k + 3] += 1.5
        out.append(
            SubjectRecord(
                subject_id=f"synth:{i:03d}",
                database="synthetic",
                signal=_zscore(sig),
                fs=fs,
            )
        )
    return out


def load_subjects(cfg: dict, database: str | None = None) -> list[SubjectRecord]:
    data_cfg = cfg.get("data", {})
    database = database or data_cfg.get("database", "fantasia")
    records: list[SubjectRecord] = []

    if database.lower() == "combined":
        dbs = list(data_cfg.get("databases", []))
        for db in dbs:
            records.extend(list(iter_database_records(db, cfg)))
    else:
        records.extend(list(iter_database_records(database, cfg)))

    if not records and data_cfg.get("allow_synthetic", True):
        records = synthetic_subjects(
            n_subjects=8,
            fs=float(data_cfg.get("target_fs", 360.0)),
            minutes=min(2.0, float(data_cfg.get("max_minutes", 30.0))),
            seed=int(cfg.get("train", {}).get("seed", 42)),
        )
    return records


class SCFImageDataset(Dataset):
    """Dataset of (SCF image, subject_label) pairs."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
    ):
        assert images.ndim == 4  # N,1,H,W
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        import torch

        x = torch.from_numpy(self.images[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def build_scf_arrays(
    subjects: list[SubjectRecord],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Segment → SCF → stack. Returns images (N,1,H,W), labels, class_names."""
    data_cfg = cfg.get("data", {})
    seg_sec = float(data_cfg.get("segment_sec", 2.0))
    img_size = int(cfg.get("model", {}).get("image_size", 128))
    max_seg = data_cfg.get("max_segments_per_subject")
    normalize = bool(cfg.get("scf", {}).get("normalize_image", True))
    scf_method = str(cfg.get("scf", {}).get("method", "caf_fft"))

    # Group by subject_id (PTB / MITDB merges already applied upstream)
    by_subj: dict[str, list[np.ndarray]] = {}
    for rec in subjects:
        segs = blind_segments(rec.signal, rec.fs, seg_sec)
        if max_seg is not None:
            segs = segs[: int(max_seg)]
        if not segs:
            continue
        by_subj.setdefault(rec.subject_id, []).extend(segs)

    class_names = sorted(by_subj.keys())
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    images: list[np.ndarray] = []
    labels: list[int] = []
    for name in class_names:
        for seg in by_subj[name]:
            img = spectral_correlation_image(
                seg,
                out_size=img_size,
                normalize=normalize,
                method=scf_method,
            )
            images.append(img[None, ...])  # 1,H,W
            labels.append(name_to_idx[name])

    if not images:
        raise RuntimeError("No SCF segments produced — check dataset paths / allow_synthetic")

    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, class_names
