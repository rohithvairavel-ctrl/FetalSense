"""Config loading and flag wiring."""

from __future__ import annotations

from pathlib import Path

import yaml

from fetalsense.models.fetalsense import build_model


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "default.yaml"


def test_default_yaml_loads():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    assert cfg["sample_rate"] == 250
    assert cfg["window_samples"] == 512
    assert cfg["patch_size"] == 16
    assert cfg["d_model"] == 128
    assert cfg["n_layers"] == 6
    assert cfg["n_heads"] == 4
    assert "use_sqi" in cfg
    assert cfg["backbone"] in ("transformer", "cnn", "bilstm")
    assert "paths" in cfg and "data_root" in cfg["paths"]
    assert cfg["ssl"]["temperature"] == 0.1
    assert cfg["peak"]["refractory_ms"] == 180
    assert 50 in cfg["metrics"]["tolerances_ms"]
    assert 100 in cfg["metrics"]["tolerances_ms"]


def test_build_model_respects_flags():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg = {**cfg, "n_layers": 2, "backbone": "cnn", "use_sqi": False}
    model = build_model(cfg)
    assert model.backbone_name == "cnn"
    assert model.use_sqi is False
