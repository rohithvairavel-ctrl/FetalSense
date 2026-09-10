"""Smoke tests: real torch forward passes on CPU for all backbones."""

from __future__ import annotations

import torch

from fetalsense.models.fetalsense import FetalSense, build_model
from fetalsense.ssl.contrastive import NTXentLoss, make_contrastive_views
from fetalsense.train.losses import FetalSenseCriterion


def _batch(B=2, C=4, T=512):
    x = torch.randn(B, C, T)
    mask = torch.ones(B, C)
    y = torch.randn(B, T) * 0.1
    p = torch.rand(B, T)
    q = torch.rand(B, C)
    return {"x": x, "channel_mask": mask, "y": y, "p": p, "q": q}


def test_transformer_forward_shapes():
    model = FetalSense(backbone="transformer", use_sqi=True)
    model.eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch["x"], batch["channel_mask"])
    assert out["y_hat"].shape == (2, 512)
    assert out["p_hat"].shape == (2, 512)
    assert out["q"].shape == (2, 4)
    assert out["z"].ndim == 2
    assert torch.isfinite(out["y_hat"]).all()
    assert ((out["p_hat"] >= 0) & (out["p_hat"] <= 1)).all()


def test_cnn_and_bilstm_forward():
    batch = _batch()
    for bb in ("cnn", "bilstm"):
        model = FetalSense(backbone=bb, use_sqi=True)
        model.eval()
        with torch.no_grad():
            out = model(batch["x"], batch["channel_mask"])
        assert out["y_hat"].shape == (2, 512)
        assert out["p_hat"].shape == (2, 512)


def test_sqi_ablation_off():
    model = FetalSense(use_sqi=False, backbone="transformer")
    batch = _batch()
    with torch.no_grad():
        out = model(batch["x"])
    assert torch.allclose(out["q"], torch.ones_like(out["q"]))


def test_ssl_and_finetune_losses_backward():
    model = FetalSense(backbone="transformer", use_sqi=True)
    batch = _batch(B=4)
    v1, v2 = make_contrastive_views(batch["x"], batch["channel_mask"])
    o1 = model(v1, batch["channel_mask"])
    o2 = model(v2, batch["channel_mask"])
    ssl = NTXentLoss(temperature=0.1)(o1["z"], o2["z"])
    assert torch.isfinite(ssl)
    ssl.backward()

    model.zero_grad(set_to_none=True)
    out = model(batch["x"], batch["channel_mask"])
    crit = FetalSenseCriterion()
    losses = crit(out, batch)
    assert "total" in losses
    losses["total"].backward()
    # At least one param got a grad
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_build_model_from_cfg():
    cfg = {
        "n_channels": 4,
        "window_samples": 512,
        "patch_size": 16,
        "d_model": 128,
        "n_layers": 2,  # lighter for speed
        "n_heads": 4,
        "use_sqi": True,
        "backbone": "transformer",
        "ssl": {"proj_dim": 64},
    }
    # Override layers via direct ctor for speed — build_model uses cfg n_layers
    model = build_model({**cfg, "n_layers": 2})
    assert isinstance(model, FetalSense)
    out = model(torch.randn(1, 4, 512))
    assert out["y_hat"].shape[-1] == 512
