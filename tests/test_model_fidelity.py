"""Unit tests for SCF image + CNN shapes (paper Table 3 / Eq. 10–11)."""

from __future__ import annotations

import numpy as np
import torch

from ecg_scf.features.scf import spectral_correlation_image
from ecg_scf.models.cnn import SpectralCNN
from ecg_scf.metrics.id_metrics import identification_rate, cmc_curve, far_frr_from_scores


def test_scf_caf_fft_shape():
    seg = np.random.randn(720).astype(np.float64)
    img = spectral_correlation_image(seg, out_size=128, method="caf_fft")
    assert img.shape == (128, 128)
    assert img.dtype == np.float32
    assert float(img.max()) <= 1.0 + 1e-5


def test_scf_bifrequency_shape():
    seg = np.random.randn(720).astype(np.float64)
    img = spectral_correlation_image(seg, out_size=64, method="bifrequency")
    assert img.shape == (64, 64)


def test_arch_a_forward():
    m = SpectralCNN(num_classes=10, arch="arch_a", image_size=128)
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    assert y.shape == (2, 10)
    # Fig. 5: single Dropout module
    assert isinstance(m.dropout, torch.nn.Dropout)


def test_arch_b_forward():
    m = SpectralCNN(num_classes=7, arch="arch_b", image_size=128)
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    assert y.shape == (2, 7)
    assert m.fc_dim == 96


def test_idr_cmc_far_frr():
    y = np.array([0, 1, 2, 0])
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert identification_rate(y, probs.argmax(1)) == 1.0
    cmc = cmc_curve(y, probs, max_rank=3)
    assert cmc[0] == 1.0
    far, frr = far_frr_from_scores(y, probs, threshold=None)
    assert far >= 0.0 and frr >= 0.0
