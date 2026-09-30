"""Spectral Correlation Function imaging (Sec. 4.3, Eq. 10–11)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def _resize_square(img: np.ndarray, out_size: int) -> np.ndarray:
    h, w = img.shape
    if h == out_size and w == out_size:
        return img
    img = zoom(img, (out_size / h, out_size / w), order=1)
    return img[:out_size, :out_size]


def spectral_correlation_image(
    segment: np.ndarray,
    out_size: int = 128,
    normalize: bool = True,
    method: str = "caf_fft",
) -> np.ndarray:
    """Build a 2D SCF magnitude image from a 1D ECG segment.

    Paper Sec. 4.3.1–4.3.2:
      Eq. 10 — cyclic autocorrelation (CAF) R^α_x(τ)
      Eq. 11 — SCF S^α_x(f) = Fourier transform of CAF over τ

    Default ``method='caf_fft'``: discrete CAF via FFT in time for each lag τ,
    then FFT along τ (Eq. 10→11). Resize to ``out_size``×``out_size`` (Sec. 5.2).

    ``method='bifrequency'``: |X(f1) X*(f2)| shortcut (legacy / faster).

    Returns:
        float32 array shaped (H, W), normalized to ~[0, 1] if ``normalize``.
    """
    x = np.asarray(segment, dtype=np.float64).reshape(-1)
    if x.size < 8:
        raise ValueError(f"segment too short: {x.size}")
    x = x - x.mean()
    n = int(x.size)
    method = (method or "caf_fft").lower()

    if method in ("bifrequency", "outer", "fft_outer"):
        X = np.fft.fft(x, n=n)
        scf = np.abs(np.outer(X, np.conj(X)))
        scf = np.fft.fftshift(scf)
    else:
        # Eq. 10: CAF(α,τ) = FFT_t[ x(t) x(t−τ) ] / N
        # Eq. 11: SCF = FFT_τ[ CAF ]
        # Vectorized for typical 2 s @ 360 Hz (n≈720); loop for long n.
        if n <= 2048:
            idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
            prod = x[idx] * x[None, :]  # (τ, t)
            caf = (np.fft.fft(prod, axis=1) / n).T  # (α, τ)
            scf = np.abs(np.fft.fft(caf, axis=1))
            scf = np.fft.fftshift(scf, axes=(0, 1))
        else:
            caf = np.empty((n, n), dtype=np.complex128)
            for tau in range(n):
                caf[:, tau] = np.fft.fft(x * np.roll(x, tau)) / n
            scf = np.abs(np.fft.fft(caf, axis=1))
            scf = np.fft.fftshift(scf, axes=(0, 1))

    img = _resize_square(
        np.nan_to_num(scf.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        out_size,
    )
    if normalize:
        m = float(np.nanmax(img)) if img.size else 0.0
        if not np.isfinite(m) or m <= 0.0:
            return np.zeros((out_size, out_size), dtype=np.float32)
        img = img / (m + 1e-12)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
