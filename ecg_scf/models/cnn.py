"""CNN Arch A / Arch B (Sec. 4.4, Fig. 5, Table 3)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv → BatchNorm → ReLU (Fig. 5)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        # Padding not stated in paper (D4); same-pad keeps 128 until MaxPool.
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpectralCNN(nn.Module):
    """2D CNN on SCF images (Fig. 5, Table 3).

    Arch A (preferred): Conv@16 → BN → ReLU → Conv@32 → BN → ReLU →
        MaxPool 2×2 → Dropout(0.3) → FC 256 → Softmax(#subjects).
    Arch B: + Conv@64 → BN → ReLU; FC 96 instead of 256.
    """

    def __init__(
        self,
        num_classes: int,
        arch: str = "arch_a",
        dropout: float = 0.3,
        in_channels: int = 1,
        image_size: int = 128,
    ):
        super().__init__()
        arch = arch.lower().replace("-", "_")
        self.arch = arch
        self.num_classes = num_classes

        layers: list[nn.Module] = [
            ConvBNReLU(in_channels, 16, 5),
            ConvBNReLU(16, 32, 5),
        ]
        if arch in ("arch_b", "b", "archb"):
            layers.append(ConvBNReLU(32, 64, 5))
            fc_dim = 96  # Table 3 FC 2
            last_ch = 64
        else:
            fc_dim = 256  # Table 3 FC 1
            last_ch = 32

        self.features = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = self.pool(self.features(dummy))
            flat = int(h.numel())

        # Fig. 5: MaxPool → Dropout → FC → Softmax (single dropout)
        self.fc = nn.Linear(flat, fc_dim)
        self.classifier = nn.Linear(fc_dim, num_classes)
        self.fc_dim = fc_dim
        self.last_ch = last_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        h = self.dropout(h)
        h = self.fc(h)
        return self.classifier(h)


def build_model(num_classes: int, cfg: dict) -> SpectralCNN:
    m = cfg.get("model", {})
    return SpectralCNN(
        num_classes=num_classes,
        arch=m.get("name", "arch_a"),
        dropout=float(m.get("dropout", 0.3)),
        in_channels=int(m.get("in_channels", 1)),
        image_size=int(m.get("image_size", 128)),
    )
