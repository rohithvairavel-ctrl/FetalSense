"""Stage 4 — multi-task fine-tune (extraction + QRS + optional SQI)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from fetalsense.data.base import synthetic_batch
from fetalsense.models.fetalsense import build_model
from fetalsense.train.losses import FetalSenseCriterion
from fetalsense.utils.seed import set_seed


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def finetune_one_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: FetalSenseCriterion,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    model.train()
    x = batch["x"]
    mask = batch.get("channel_mask")
    outputs = model(x, mask)
    losses = criterion(outputs, batch)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FetalSense fine-tune")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional SSL encoder weights")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state), strict=False)
        print(f"Loaded checkpoint {args.checkpoint}")

    loss_cfg = cfg.get("loss", {})
    criterion = FetalSenseCriterion(
        lambda_qrs=float(loss_cfg.get("lambda_qrs", 1.0)),
        lambda_ext=float(loss_cfg.get("lambda_ext", 0.5)),
        lambda_sqi=float(loss_cfg.get("lambda_sqi", 0.1)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        pearson_weight=float(loss_cfg.get("pearson_weight", 1.0)),
        mse_weight=float(loss_cfg.get("mse_weight", 0.5)),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("finetune", {}).get("head_lr", 3e-4)),
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 0.01)),
    )

    if args.steps > 0:
        for step in range(args.steps):
            batch = synthetic_batch(seed=step)
            batch = {k: v.to(device) for k, v in batch.items()}
            stats = finetune_one_step(model, batch, criterion, opt)
            print(f"step={step} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
    else:
        print(
            "Fine-tune entrypoint ready. Pass --steps N for synthetic dry-run, "
            "or attach Challenge-2013 / ADFECGDB DataLoaders."
        )


if __name__ == "__main__":
    main()
