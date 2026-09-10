"""CNN and BiLSTM backbones for Transformer ablation."""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """1D CNN encoder producing a token sequence of length N ≈ T/P."""

    def __init__(
        self,
        n_channels: int = 4,
        d_model: int = 128,
        patch_size: int = 16,
        depth: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_channels
        for i in range(depth):
            out_ch = d_model
            stride = 2 if i < 4 else 1  # roughly /16 over 4 strides
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → tokens (B, N, d)."""
        h = self.net(x)
        h = self.proj(h)
        return h.transpose(1, 2)  # (B, N, d)


class BiLSTMBackbone(nn.Module):
    """BiLSTM over fused patches for ablation."""

    def __init__(
        self,
        n_channels: int = 4,
        d_model: int = 128,
        patch_size: int = 16,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(n_channels * patch_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, N, d)."""
        B, C, T = x.shape
        P = self.patch_size
        if C < self.n_channels:
            pad = torch.zeros(B, self.n_channels - C, T, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif C > self.n_channels:
            x = x[:, : self.n_channels]
        patches = x.unfold(2, P, P)  # (B, C, N, P)
        N = patches.shape[2]
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        tokens = self.patch_proj(patches)
        out, _ = self.lstm(tokens)
        return out
