"""CLOCS-inspired NT-Xent contrastive loss with temporal + channel views."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy (InfoNCE) for paired views.

    Expects embeddings z1, z2 of shape (B, D) from two augmented views of the
    same batch of windows. Temperature τ default 0.1 (locked).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        # Mask self-similarities
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)

        # Positives: i <-> i+B
        pos = torch.cat(
            [torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)],
            dim=0,
        )
        loss = F.cross_entropy(sim, pos)
        return loss


def make_contrastive_views(
    x: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create temporal and channel-dropout views (CLOCS-like, abdomen-adapted).

    View T: mild random time shift (circular) + noise.
    View C: random channel dropout / amplitude scale + noise.
    """
    B, C, T = x.shape
    device = x.device

    # Temporal view: circular shift ± up to 8 samples + noise
    shifts = torch.randint(-8, 9, (B,), device=device)
    v_t = torch.stack([torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1) for i in range(B)], dim=0)
    v_t = v_t + 0.01 * torch.randn_like(v_t)

    # Channel view: drop ~1 channel (keep at least 1), scale amplitude
    v_c = x.clone()
    keep = torch.ones(B, C, device=device)
    for i in range(B):
        n_drop = 1 if C > 1 else 0
        if n_drop:
            drop_idx = torch.randperm(C, device=device)[:n_drop]
            keep[i, drop_idx] = 0.0
    if channel_mask is not None:
        keep = keep * channel_mask.float()
    v_c = v_c * keep.unsqueeze(-1)
    scale = 0.8 + 0.4 * torch.rand(B, 1, 1, device=device)
    v_c = v_c * scale + 0.01 * torch.randn_like(v_c)
    return v_t, v_c
