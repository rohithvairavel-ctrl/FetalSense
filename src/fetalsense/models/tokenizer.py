"""Patch tokenizer: fused multi-lead patches → d-dim tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """Non-overlapping (or strided) fused multi-lead patch embedding.

    Flatten each (C × P) patch and linearly project to d_model.
    """

    def __init__(
        self,
        n_channels: int = 4,
        patch_size: int = 16,
        d_model: int = 128,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.proj = nn.Linear(n_channels * patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))  # max tokens buffer
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) SQI-weighted (or raw) multi-lead window.

        Returns:
            tokens: (B, N, d) where N = floor((T - P) / stride) + 1
        """
        B, C, T = x.shape
        P, S = self.patch_size, self.stride
        if C != self.n_channels:
            # Pad or crop channels to expected C
            if C < self.n_channels:
                pad = torch.zeros(B, self.n_channels - C, T, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            else:
                x = x[:, : self.n_channels]
            C = self.n_channels

        # Unfold along time: (B, C, N, P)
        patches = x.unfold(dimension=2, size=P, step=S)  # (B, C, N, P)
        N = patches.shape[2]
        patches = patches.permute(0, 2, 1, 3).contiguous()  # (B, N, C, P)
        patches = patches.view(B, N, C * P)
        tokens = self.proj(patches)  # (B, N, d)
        pos = self.pos_embed[:, :N, :]
        return tokens + pos
