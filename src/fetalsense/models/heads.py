"""Task heads: QRS likelihood and fetal waveform extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnConvDecoder(nn.Module):
    """Cross-attention once over encoder memory, then ConvTranspose upsample → T samples."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        patch_size: int = 16,
        out_channels: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # Progressive upsample: each ConvTranspose doubles length (patch tokens → T)
        # For P=16: need 4× upsample by 2
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                )
                for _ in range(4)  # 16x
            ]
        )
        self.out_conv = nn.Conv1d(d_model, out_channels, kernel_size=3, padding=1)

    def forward(self, memory: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            memory: (B, N, d) encoder tokens (patch tokens only preferred).
            target_len: desired waveform length T.

        Returns:
            y_hat: (B, T) extracted fetal ECG.
        """
        B, N, D = memory.shape
        # One learned query per patch position via linear proj of memory (identity path)
        # Cross-attn: use memory as Q/K/V with residual (self) then treat as sequence features
        q = memory  # (B, N, d) — attend within memory once (cross to itself as locked hybrid)
        h, _ = self.cross_attn(q, memory, memory, need_weights=False)
        h = self.norm(memory + h)  # (B, N, d)
        h = h.transpose(1, 2)  # (B, d, N)
        for up in self.ups:
            h = up(h)
        # Match exact target length
        if h.shape[-1] != target_len:
            h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        y = self.out_conv(h).squeeze(1)  # (B, T)
        return y


class ExtractionHead(nn.Module):
    """Wrapper around cross-attn conv decoder → waveform."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, patch_size: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.decoder = CrossAttnConvDecoder(d_model, n_heads, patch_size, dropout=dropout)

    def forward(self, memory: torch.Tensor, target_len: int) -> torch.Tensor:
        return self.decoder(memory, target_len)


class QRSHead(nn.Module):
    """Temporal QRS likelihood via upsample of token stream → Conv1d → sigmoid."""

    def __init__(self, d_model: int = 128, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        self.conv = nn.Conv1d(d_model, 1, kernel_size=5, padding=2)

    def forward(self, memory: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            memory: (B, N, d) patch tokens.
            target_len: T.

        Returns:
            p_hat: (B, T) in (0, 1).
        """
        h = memory.transpose(1, 2)  # (B, d, N)
        for up in self.ups:
            h = up(h)
        if h.shape[-1] != target_len:
            h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        logits = self.conv(h).squeeze(1)
        return torch.sigmoid(logits)
