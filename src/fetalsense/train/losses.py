"""Fine-tune loss terms: QRS BCE(+Dice), Pearson+MSE extraction, optional SQI BCE."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - Pearson correlation, averaged over batch. pred/target: (B, T)."""
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    tgt_c = target - target.mean(dim=-1, keepdim=True)
    num = (pred_c * tgt_c).sum(dim=-1)
    den = pred_c.norm(dim=-1) * tgt_c.norm(dim=-1) + eps
    corr = num / den
    return (1.0 - corr).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on probability maps (B, T)."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    inter = (pred * target).sum(dim=-1)
    union = pred.sum(dim=-1) + target.sum(dim=-1)
    dice = (2 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


class FetalSenseCriterion(nn.Module):
    def __init__(
        self,
        lambda_qrs: float = 1.0,
        lambda_ext: float = 0.5,
        lambda_sqi: float = 0.1,
        dice_weight: float = 0.5,
        pearson_weight: float = 1.0,
        mse_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_qrs = lambda_qrs
        self.lambda_ext = lambda_ext
        self.lambda_sqi = lambda_sqi
        self.dice_weight = dice_weight
        self.pearson_weight = pearson_weight
        self.mse_weight = mse_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        p_hat = outputs["p_hat"]
        y_hat = outputs["y_hat"]
        q = outputs["q"]

        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=p_hat.device)

        # QRS
        if "p" in batch:
            bce = F.binary_cross_entropy(p_hat.clamp(1e-6, 1 - 1e-6), batch["p"])
            dice = dice_loss(p_hat, batch["p"])
            l_qrs = bce + self.dice_weight * dice
            # Quality-aware weighting
            q_weight = q.mean(dim=-1).detach().clamp_min(0.1).mean()
            l_qrs = l_qrs * q_weight
            losses["qrs"] = l_qrs
            total = total + self.lambda_qrs * l_qrs

        # Extraction
        if "y" in batch:
            l_pear = pearson_loss(y_hat, batch["y"])
            l_mse = F.mse_loss(y_hat, batch["y"])
            l_ext = self.pearson_weight * l_pear + self.mse_weight * l_mse
            losses["ext"] = l_ext
            total = total + self.lambda_ext * l_ext

        # SQI supervision (optional)
        if "q" in batch and self.lambda_sqi > 0:
            l_sqi = F.binary_cross_entropy(q.clamp(1e-6, 1 - 1e-6), batch["q"])
            losses["sqi"] = l_sqi
            total = total + self.lambda_sqi * l_sqi

        losses["total"] = total
        return losses
