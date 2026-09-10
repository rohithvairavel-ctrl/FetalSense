"""Evaluation: peak-pick + Sens/PPV/F1 at 50 ms and 100 ms tolerances."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from fetalsense.data.base import synthetic_batch
from fetalsense.metrics.fqrs import score_probability_map
from fetalsense.models.fetalsense import build_model
from fetalsense.utils.peaks import pick_peaks
from fetalsense.utils.seed import set_seed


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    model.eval()
    out = model(batch["x"], batch.get("channel_mask"))
    fs = float(cfg.get("sample_rate", 250))
    peak_cfg = cfg.get("peak", {})
    tol_list = tuple(cfg.get("metrics", {}).get("tolerances_ms", [50, 100]))
    results = []
    p_hat = out["p_hat"].cpu().numpy()
    for i in range(p_hat.shape[0]):
        # Prefer soft-target peaks as pseudo-ref when available
        if "p" in batch:
            ref = pick_peaks(
                batch["p"][i].cpu().numpy(),
                fs=fs,
                threshold=float(peak_cfg.get("threshold", 0.3)),
                refractory_ms=float(peak_cfg.get("refractory_ms", 180)),
            )
        else:
            ref = np.array([], dtype=np.int64)
        scores = score_probability_map(
            p_hat[i],
            ref,
            fs=fs,
            tolerances_ms=tol_list,
            threshold=float(peak_cfg.get("threshold", 0.3)),
            refractory_ms=float(peak_cfg.get("refractory_ms", 180)),
        )
        results.append(scores)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FetalSense eval")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on one synthetic batch")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)

    if args.synthetic or True:
        batch = {k: v.to(device) for k, v in synthetic_batch(batch_size=2, seed=0).items()}
        results = evaluate_batch(model, batch, cfg)
        for i, r in enumerate(results):
            print(f"sample[{i}]: {r}")


if __name__ == "__main__":
    main()
