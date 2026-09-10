"""FetalSense end-to-end model: SQI → backbone → extraction + QRS heads."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from fetalsense.models.cnn_baseline import BiLSTMBackbone, CNNBackbone
from fetalsense.models.heads import ExtractionHead, QRSHead
from fetalsense.models.sqi import SQIModule
from fetalsense.models.tokenizer import PatchTokenizer
from fetalsense.models.transformer import TransformerEncoder


class FetalSense(nn.Module):
    """Quality-aware multi-task fetal ECG model.

    Forward returns dict with keys: y_hat, p_hat, q, tokens, embedding.
    """

    def __init__(
        self,
        n_channels: int = 4,
        window_samples: int = 512,
        patch_size: int = 16,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 4,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        use_sqi: bool = True,
        sqi_mode: Literal["classic", "cnn", "both"] = "classic",
        backbone: Literal["transformer", "cnn", "bilstm"] = "transformer",
        quality_tokens: bool = True,
        soft_channel_weights: bool = True,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_sqi = use_sqi
        self.backbone_name = backbone
        self.quality_tokens_flag = quality_tokens
        self.soft_channel_weights = soft_channel_weights

        self.sqi = SQIModule(d_model=d_model, mode=sqi_mode, n_channels=n_channels)

        if backbone == "transformer":
            self.tokenizer = PatchTokenizer(n_channels, patch_size, d_model)
            self.encoder = TransformerEncoder(d_model, n_layers, n_heads, ffn_mult, dropout)
            self.cnn_backbone = None
            self.bilstm_backbone = None
        elif backbone == "cnn":
            self.tokenizer = None
            self.encoder = None
            self.cnn_backbone = CNNBackbone(n_channels, d_model, patch_size, depth=n_layers, dropout=dropout)
            self.bilstm_backbone = None
        elif backbone == "bilstm":
            self.tokenizer = None
            self.encoder = None
            self.cnn_backbone = None
            self.bilstm_backbone = BiLSTMBackbone(n_channels, d_model, patch_size, n_layers=max(2, n_layers // 2), dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.extraction = ExtractionHead(d_model, n_heads, patch_size, dropout)
        self.qrs = QRSHead(d_model, patch_size)

        # SSL projection head
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, proj_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SQI + backbone → patch memory tokens, quality, embedding."""
        B, C, T = x.shape
        if self.use_sqi:
            q, x_w, q_tokens = self.sqi(x, channel_mask)
            x_in = x_w if self.soft_channel_weights else x
        else:
            q = torch.ones(B, C, device=x.device, dtype=x.dtype)
            q_tokens = self.sqi.quality_proj(q.unsqueeze(-1))
            x_in = x

        if self.backbone_name == "transformer":
            assert self.tokenizer is not None and self.encoder is not None
            tokens = self.tokenizer(x_in)
            if self.use_sqi and self.quality_tokens_flag:
                tokens = torch.cat([q_tokens, tokens], dim=1)
            memory = self.encoder(tokens)
            # Strip quality tokens for temporal heads
            if self.use_sqi and self.quality_tokens_flag:
                patch_mem = memory[:, C:, :]
            else:
                patch_mem = memory
        elif self.backbone_name == "cnn":
            assert self.cnn_backbone is not None
            patch_mem = self.cnn_backbone(x_in)
            memory = patch_mem
        else:
            assert self.bilstm_backbone is not None
            patch_mem = self.bilstm_backbone(x_in)
            memory = patch_mem

        # Global embedding for SSL (mean pool)
        embedding = patch_mem.mean(dim=1)
        return patch_mem, q, embedding

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T) multi-channel abdominal ECG.
            channel_mask: optional (B, C).

        Returns:
            dict with y_hat (B,T), p_hat (B,T), q (B,C), tokens (B,N,d), z (B, proj_dim)
        """
        T = x.shape[-1]
        patch_mem, q, embedding = self.encode(x, channel_mask)
        y_hat = self.extraction(patch_mem, T)
        p_hat = self.qrs(patch_mem, T)
        z = self.proj(embedding)
        return {
            "y_hat": y_hat,
            "p_hat": p_hat,
            "q": q,
            "tokens": patch_mem,
            "embedding": embedding,
            "z": z,
        }


def build_model(cfg: Dict[str, Any]) -> FetalSense:
    """Construct FetalSense from a nested config dict (YAML-loaded)."""
    return FetalSense(
        n_channels=int(cfg.get("n_channels", 4)),
        window_samples=int(cfg.get("window_samples", 512)),
        patch_size=int(cfg.get("patch_size", 16)),
        d_model=int(cfg.get("d_model", 128)),
        n_layers=int(cfg.get("n_layers", 6)),
        n_heads=int(cfg.get("n_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        ffn_mult=int(cfg.get("ffn_mult", 4)),
        use_sqi=bool(cfg.get("use_sqi", True)),
        sqi_mode=cfg.get("sqi_mode", "classic"),
        backbone=cfg.get("backbone", "transformer"),
        quality_tokens=bool(cfg.get("quality_tokens", True)),
        soft_channel_weights=bool(cfg.get("soft_channel_weights", True)),
        proj_dim=int(cfg.get("ssl", {}).get("proj_dim", 128)),
    )
