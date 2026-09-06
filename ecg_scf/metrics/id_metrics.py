"""Identification metrics: IDR, CMC, FAR, FRR (Sec. 4.5)."""

from __future__ import annotations

import numpy as np


def identification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank-1 IDR / accuracy (Sec. 4.5)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def cmc_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    max_rank: int = 10,
) -> np.ndarray:
    """Cumulative match characteristic from softmax scores (N, C)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs)
    n, c = probs.shape
    max_rank = int(min(max_rank, c))
    order = np.argsort(-probs, axis=1)
    hits = np.zeros(max_rank, dtype=np.float64)
    for r in range(max_rank):
        top = order[:, : r + 1]
        hits[r] = np.mean([y_true[i] in top[i] for i in range(n)])
    return hits


def far_frr_from_scores(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
) -> tuple[float, float]:
    """FAR / FRR from softmax posteriors (Sec. 4.5; operating point D2).

    Verification-style scores on closed-set probes:
      - Genuine score = P(true class)
      - Impostor scores = P(other classes)

    If ``threshold`` is None, use the equal-error operating point (EER-like)
    on pooled genuine/impostor scores; else apply the given threshold.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    n, c = probs.shape
    if n == 0 or c == 0:
        return 0.0, 0.0

    genuine = probs[np.arange(n), y_true]
    impostor_list = []
    for i in range(n):
        mask = np.ones(c, dtype=bool)
        mask[y_true[i]] = False
        impostor_list.append(probs[i, mask])
    impostor = np.concatenate(impostor_list) if impostor_list else np.array([], dtype=np.float64)

    if threshold is None:
        # Sweep for approximate EER
        scores = np.concatenate([genuine, impostor]) if impostor.size else genuine
        thr_candidates = np.unique(scores)
        if thr_candidates.size == 0:
            return 0.0, 0.0
        best_gap = 1e9
        best_far, best_frr = 0.0, 0.0
        for t in thr_candidates:
            frr = float(np.mean(genuine < t)) if genuine.size else 0.0
            far = float(np.mean(impostor >= t)) if impostor.size else 0.0
            gap = abs(far - frr)
            if gap < best_gap:
                best_gap = gap
                best_far, best_frr = far, frr
        return best_far, best_frr

    t = float(threshold)
    frr = float(np.mean(genuine < t)) if genuine.size else 0.0
    far = float(np.mean(impostor >= t)) if impostor.size else 0.0
    return far, frr


def summarize_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = 0.5,
    max_rank: int = 10,
    far_frr_mode: str = "eer",
) -> dict:
    """Aggregate IDR / FAR / FRR / CMC.

    ``far_frr_mode``: ``eer`` ignores threshold and uses EER-like point;
    ``threshold`` uses ``threshold`` on genuine/impostor scores.
    """
    y_pred = probs.argmax(axis=1)
    idr = identification_rate(y_true, y_pred)
    cmc = cmc_curve(y_true, probs, max_rank=max_rank)
    thr = None if far_frr_mode == "eer" else threshold
    far, frr = far_frr_from_scores(y_true, probs, threshold=thr)
    return {
        "idr": idr,
        "far": far,
        "frr": frr,
        "cmc": cmc.tolist(),
        "far_frr_mode": far_frr_mode,
    }
