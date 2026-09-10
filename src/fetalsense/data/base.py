"""Base dataset interfaces for windowed multi-lead abdominal ECG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowSample:
    """Single training / eval window."""

    x: np.ndarray  # (C, T)
    y: Optional[np.ndarray] = None  # (T,) fetal waveform GT
    p: Optional[np.ndarray] = None  # (T,) soft QRS targets
    q: Optional[np.ndarray] = None  # (C,) quality weak labels
    channel_mask: Optional[np.ndarray] = None  # (C,)
    meta: Optional[Dict[str, Any]] = None


class FetalECGWindowDataset(Dataset):
    """In-memory list of windows; subclasses populate `self.samples`."""

    def __init__(self, samples: Optional[List[WindowSample]] = None) -> None:
        self.samples: List[WindowSample] = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        C, T = s.x.shape
        out: Dict[str, Any] = {
            "x": torch.from_numpy(s.x.astype(np.float32)),
            "channel_mask": torch.from_numpy(
                (s.channel_mask if s.channel_mask is not None else np.ones(C, dtype=np.float32)).astype(np.float32)
            ),
        }
        if s.y is not None:
            out["y"] = torch.from_numpy(s.y.astype(np.float32))
        if s.p is not None:
            out["p"] = torch.from_numpy(s.p.astype(np.float32))
        if s.q is not None:
            out["q"] = torch.from_numpy(s.q.astype(np.float32))
        if s.meta is not None:
            out["meta"] = s.meta
        return out


def collate_windows(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors; skip non-tensor keys like meta."""
    keys = [k for k in batch[0] if k != "meta" and isinstance(batch[0][k], torch.Tensor)]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    if "meta" in batch[0]:
        out["meta"] = [b.get("meta") for b in batch]
    return out


def synthetic_batch(
    batch_size: int = 2,
    n_channels: int = 4,
    length: int = 512,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Deterministic synthetic batch for smoke tests / dry-runs."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch_size, n_channels, length)).astype(np.float32)
    y = rng.standard_normal((batch_size, length)).astype(np.float32) * 0.1
    t = np.linspace(0, 1, length, dtype=np.float32)
    p = (0.5 * (1 + np.sin(2 * np.pi * 2.5 * t))).astype(np.float32)
    p = np.broadcast_to(p, (batch_size, length)).copy()
    q = rng.uniform(0.5, 1.0, size=(batch_size, n_channels)).astype(np.float32)
    mask = np.ones((batch_size, n_channels), dtype=np.float32)
    return {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "p": torch.from_numpy(p),
        "q": torch.from_numpy(q),
        "channel_mask": torch.from_numpy(mask),
    }
