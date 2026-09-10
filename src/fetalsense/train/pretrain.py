"""Stage 3 — SSL pretraining with CLOCS-like NT-Xent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from fetalsense.data.base import synthetic_batch
from fetalsense.models.fetalsense import build_model
from fetalsense.ssl.contrastive import NTXentLoss, make_contrastive_views
from fetalsense.utils.seed import set_seed


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def pretrain_one_step(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    channel_mask: torch.Tensor | None,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train()
    v1, v2 = make_contrastive_views(batch_x, channel_mask)
    o1 = model(v1, channel_mask)
    o2 = model(v2, channel_mask)
    loss = criterion(o1["z"], o2["z"])
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(loss.detach().cpu())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FetalSense SSL pretrain")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=0, help="If >0, run synthetic dry-run steps")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = torch.device(args.device)

    model = build_model(cfg).to(device)
    tau = float(cfg.get("ssl", {}).get("temperature", 0.1))
    criterion = NTXentLoss(temperature=tau)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("pretrain", {}).get("lr", 1e-3)),
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 0.01)),
    )

    if args.steps > 0:
        for step in range(args.steps):
            batch = synthetic_batch(
                batch_size=min(4, int(cfg.get("pretrain", {}).get("batch_size", 64))),
                n_channels=int(cfg.get("n_channels", 4)),
                length=int(cfg.get("window_samples", 512)),
                seed=step,
            )
            x = batch["x"].to(device)
            mask = batch["channel_mask"].to(device)
            loss = pretrain_one_step(model, x, mask, criterion, opt)
            print(f"step={step} loss={loss:.4f}")
    else:
        print(
            "SSL pretrain entrypoint ready. Pass --steps N for a synthetic dry-run, "
            "or wire a DataLoader over unlabeled pools (FECGSYNDB / NInFEA / NIFECGDB)."
        )
        ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "ssl_init.pt")
        print(f"Wrote initial weights to {ckpt_dir / 'ssl_init.pt'}")


if __name__ == "__main__":
    main()
