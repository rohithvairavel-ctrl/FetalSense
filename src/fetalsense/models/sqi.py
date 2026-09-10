"""Signal Quality Index: classic features → MLP and optional CNN head."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _classic_sqi_features(x: torch.Tensor) -> torch.Tensor:
    """Hand-crafted per-channel features. x: (B, C, T) → (B, C, F)."""
    # Kurtosis (excess)
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=-1).clamp_min(1e-8)
    kurt = centered.pow(4).mean(dim=-1) / (var.pow(2)) - 3.0  # (B, C)

    # Spectral power ratio via FFT energy in fetal QRS-ish band vs total
    # Approximate with high-pass energy / total energy (time-domain proxy)
    diff = x[..., 1:] - x[..., :-1]
    high_e = diff.pow(2).mean(dim=-1)
    tot_e = x.pow(2).mean(dim=-1).clamp_min(1e-8)
    spectral_ratio = high_e / tot_e

    # Baseline wander index: low-frequency energy proxy (moving mean variance)
    # Use simple downsample-then-upsample residual energy
    T = x.shape[-1]
    kernel = max(3, T // 32)
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    x_pad = F.pad(x, (pad, pad), mode="reflect")
    # depthwise avg pool via unfold mean
    unfold = x_pad.unfold(-1, kernel, 1)  # (B, C, T, K)
    baseline = unfold.mean(dim=-1)
    wander = (x - baseline).pow(2).mean(dim=-1) / tot_e

    # Amplitude std
    amp_std = x.std(dim=-1)

    feats = torch.stack([kurt, spectral_ratio, wander, amp_std], dim=-1)  # (B, C, 4)
    return feats


class ClassicSQIMLP(nn.Module):
    """MLP over classic SQI features → per-channel quality in (0, 1)."""

    def __init__(self, n_features: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = _classic_sqi_features(x)  # (B, C, F)
        q = self.mlp(feats).squeeze(-1)  # (B, C)
        return q


class CNNSQIHead(nn.Module):
    """Small 1D CNN per channel → global pool → sigmoid quality."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → process each channel independently
        B, C, T = x.shape
        h = x.reshape(B * C, 1, T)
        h = self.conv(h).squeeze(-1)  # (B*C, hidden)
        q = torch.sigmoid(self.fc(h)).view(B, C)
        return q


class SQIModule(nn.Module):
    """Quality estimation with soft weights + quality token projection.

    Modes: classic | cnn | both (average of classic and cnn).
    """

    def __init__(
        self,
        d_model: int = 128,
        mode: Literal["classic", "cnn", "both"] = "classic",
        n_channels: int = 4,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_channels = n_channels
        self.classic: Optional[ClassicSQIMLP] = None
        self.cnn: Optional[CNNSQIHead] = None
        if mode in ("classic", "both"):
            self.classic = ClassicSQIMLP()
        if mode in ("cnn", "both"):
            self.cnn = CNNSQIHead()
        self.quality_proj = nn.Linear(1, d_model)

    def forward(
        self, x: torch.Tensor, channel_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) abdominal ECG.
            channel_mask: optional (B, C) with 1=valid lead.

        Returns:
            q: (B, C) quality scores in (0, 1)
            x_weighted: (B, C, T) soft-weighted signal
            quality_tokens: (B, C, d) projected quality embeddings
        """
        qs = []
        if self.classic is not None:
            qs.append(self.classic(x))
        if self.cnn is not None:
            qs.append(self.cnn(x))
        q = qs[0] if len(qs) == 1 else 0.5 * (qs[0] + qs[1])

        if channel_mask is not None:
            q = q * channel_mask.float()

        x_weighted = x * q.unsqueeze(-1)
        quality_tokens = self.quality_proj(q.unsqueeze(-1))  # (B, C, d)
        return q, x_weighted, quality_tokens
