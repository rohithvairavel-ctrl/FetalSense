"""Stage-1: supervised fQRS prototype on Challenge-2013 Set A (record-level split)."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fetalsense.data.base import collate_windows
from fetalsense.data.challenge2013 import Challenge2013Dataset
from fetalsense.metrics.fqrs import sens_ppv_f1
from fetalsense.models.fetalsense import build_model
from fetalsense.train.losses import FetalSenseCriterion
from fetalsense.utils.peaks import pick_peaks
from fetalsense.utils.seed import set_seed

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def record_level_indices(
    dataset: Challenge2013Dataset,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[List[int], List[int], List[str], List[str]]:
    by_rec: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        rec = (s.meta or {}).get("record", f"unk_{i}")
        by_rec[rec].append(i)
    records = sorted(by_rec.keys())
    rng = np.random.default_rng(seed)
    order = records.copy()
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_frac)))
    val_recs = set(order[:n_val])
    train_recs = [r for r in order if r not in val_recs]
    val_recs_l = [r for r in order if r in val_recs]
    train_idx = [i for r in train_recs for i in by_rec[r]]
    val_idx = [i for r in val_recs_l for i in by_rec[r]]
    return train_idx, val_idx, train_recs, val_recs_l


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fs: float = 250.0,
    threshold: float = 0.3,
    refractory_ms: float = 180.0,
) -> Dict[str, Any]:
    model.eval()
    totals = {
        "50ms": {"tp": 0, "fp": 0, "fn": 0},
        "100ms": {"tp": 0, "fp": 0, "fn": 0},
    }
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch.get("channel_mask")
        if mask is not None:
            mask = mask.to(device)
        out = model(x, mask)
        p_hat = out["p_hat"].detach().cpu().numpy()
        metas: Sequence[Any] = batch.get("meta") or [{}] * len(p_hat)
        for b in range(p_hat.shape[0]):
            ref = np.asarray((metas[b] or {}).get("ref_peaks") or [], dtype=np.int64)
            det = pick_peaks(p_hat[b], fs=fs, threshold=threshold, refractory_ms=refractory_ms)
            for tol_key, tol_ms in (("50ms", 50.0), ("100ms", 100.0)):
                m = sens_ppv_f1(ref, det, fs=fs, tolerance_ms=tol_ms)
                totals[tol_key]["tp"] += int(m["tp"])
                totals[tol_key]["fp"] += int(m["fp"])
                totals[tol_key]["fn"] += int(m["fn"])

    summary: Dict[str, Any] = {}
    for k, c in totals.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) else 0.0
        summary[k] = {"sensitivity": sens, "ppv": ppv, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Stage-1 supervised fQRS on Challenge-2013")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--backbone", type=str, default="transformer")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    device = torch.device(args.device)

    data_root = Path(cfg["paths"]["challenge2013"])
    logger.info("Loading Challenge-2013 from %s", data_root)
    ds = Challenge2013Dataset(data_root, max_records=args.max_records)
    if len(ds) == 0:
        raise SystemExit(f"No windows loaded from {data_root}")

    train_idx, val_idx, train_recs, val_recs = record_level_indices(
        ds, val_frac=args.val_frac, seed=int(cfg.get("train", {}).get("seed", 42))
    )
    logger.info(
        "Records train/val=%d/%d windows train/val=%d/%d",
        len(train_recs),
        len(val_recs),
        len(train_idx),
        len(val_idx),
    )

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_windows,
        num_workers=0,
    )

    cfg = dict(cfg)
    cfg["use_sqi"] = False
    cfg["backbone"] = args.backbone
    model = build_model(cfg).to(device)
    criterion = FetalSenseCriterion(lambda_qrs=1.0, lambda_ext=0.0, lambda_sqi=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    ckpt_dir = Path(cfg["paths"].get("checkpoints", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history: List[Dict[str, Any]] = []

    peak_thr = float(cfg.get("peak", {}).get("threshold", 0.3))
    refractory = float(cfg.get("peak", {}).get("refractory_ms", 180))

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch_t = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch_t["x"], batch_t.get("channel_mask"))
            loss_dict = criterion(out, batch_t)
            loss = loss_dict["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device, threshold=peak_thr, refractory_ms=refractory)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val": metrics,
        }
        history.append(row)
        f1_50 = metrics["50ms"]["f1"]
        f1_100 = metrics["100ms"]["f1"]
        logger.info(
            "epoch %d loss=%.4f F1@50=%.4f F1@100=%.4f (sens50=%.3f ppv50=%.3f)",
            epoch,
            row["train_loss"] or 0.0,
            f1_50,
            f1_100,
            metrics["50ms"]["sensitivity"],
            metrics["50ms"]["ppv"],
        )
        if f1_50 >= best_f1:
            best_f1 = f1_50
            ckpt_path = ckpt_dir / "stage1_challenge2013.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                    "train_records": train_recs,
                    "val_records": val_recs,
                },
                ckpt_path,
            )
            logger.info("Saved best checkpoint → %s", ckpt_path)

    out_json = {
        "stage": 1,
        "dataset": "challenge-2013-set-a",
        "backbone": args.backbone,
        "use_sqi": False,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_train_records": len(train_recs),
        "n_val_records": len(val_recs),
        "n_train_windows": len(train_idx),
        "n_val_windows": len(val_idx),
        "val_records": val_recs,
        "best_val_f1_50ms": best_f1,
        "history": history,
        "best_checkpoint": str(ckpt_dir / "stage1_challenge2013.pt"),
        "note": "Prototype Stage-1 numbers only — not a paper claim. Record-level split.",
    }
    metrics_path = ckpt_dir / "stage1_metrics.json"
    metrics_path.write_text(json.dumps(out_json, indent=2))
    notes = ckpt_dir.parent / "docs" / "STAGE1_NOTES.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        f"""# Stage-1 supervised fQRS (Challenge-2013)

- **Split:** record-level {len(train_recs)}/{len(val_recs)} train/val (seed from config)
- **Model:** backbone=`{args.backbone}`, `use_sqi=false`, QRS-only loss
- **Hardware:** `{device}`
- **Epochs:** {args.epochs}
- **Best val F1@50ms:** {best_f1:.4f}
- **Checkpoint:** `checkpoints/stage1_challenge2013.pt`
- **Metrics JSON:** `checkpoints/stage1_metrics.json`

These are prototype pipeline numbers, not publication claims.
"""
    )
    print(json.dumps({"best_val_f1_50ms": best_f1, "metrics_path": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
