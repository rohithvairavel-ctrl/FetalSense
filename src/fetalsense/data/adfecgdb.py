"""ADFECGDB (Abdominal and Direct Fetal ECG Database) loader sketch.

Expected layout under ``paths.adfecgdb``::

    r01.edf / or WFDB records r01.dat + r01.hea
    ...

Direct (scalp) channel may be used as waveform GT when present.
Empty dataset if path missing — tests must not depend on downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from fetalsense.data.base import FetalECGWindowDataset, WindowSample
from fetalsense.utils.preprocess import bandpass_filter, make_soft_qrs_targets, zscore_per_channel

logger = logging.getLogger(__name__)

TARGET_FS = 250
WINDOW = 512


class ADFECGDBDataset(FetalECGWindowDataset):
    def __init__(
        self,
        root: str | Path,
        window: int = WINDOW,
        stride: int = 256,
        n_channels: int = 4,
        max_records: Optional[int] = None,
    ) -> None:
        root = Path(root)
        samples: List[WindowSample] = []
        if not root.exists():
            logger.warning("ADFECGDB path missing: %s (empty dataset)", root)
            super().__init__(samples)
            return

        try:
            import wfdb
        except ImportError:
            logger.info("wfdb not installed; ADFECGDB loader inactive.")
            super().__init__(samples)
            return

        hea_files = sorted(root.glob("*.hea"))
        if max_records is not None:
            hea_files = hea_files[:max_records]

        for hea in hea_files:
            try:
                rec = wfdb.rdrecord(str(hea.with_suffix("")))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip %s: %s", hea, exc)
                continue
            sig = np.asarray(rec.p_signal, dtype=np.float64).T
            fs = float(rec.fs)
            if abs(fs - TARGET_FS) > 1e-3:
                from math import gcd

                from scipy.signal import resample_poly

                g = gcd(int(TARGET_FS), int(fs))
                up, down = int(TARGET_FS) // g, int(fs) // g
                sig = np.stack([resample_poly(ch, up, down) for ch in sig], axis=0)

            # Heuristic: first 4 abdominal; last channel may be direct fetal
            abd = sig[:n_channels]
            direct = sig[-1] if sig.shape[0] > n_channels else None
            abd = bandpass_filter(abd, fs=TARGET_FS)
            C = abd.shape[0]
            if C < n_channels:
                pad = np.zeros((n_channels - C, abd.shape[1]))
                mask = np.array([1] * C + [0] * (n_channels - C), dtype=np.float32)
                abd = np.concatenate([abd, pad], axis=0)
            else:
                mask = np.ones(n_channels, dtype=np.float32)

            # Annotations if any
            peaks: List[int] = []
            for ext in ("qrs", "fqrs", "atr"):
                try:
                    ann = wfdb.rdann(str(hea.with_suffix("")), extension=ext)
                    peaks = [int(s) for s in ann.sample]
                    break
                except Exception:  # noqa: BLE001
                    continue

            T = abd.shape[1]
            for start in range(0, max(1, T - window + 1), stride):
                end = start + window
                if end > T:
                    break
                x = zscore_per_channel(abd[:, start:end]).astype(np.float32)
                local_peaks = [p - start for p in peaks if start <= p < end]
                p_soft = make_soft_qrs_targets(np.array(local_peaks), window).astype(np.float32)
                y = None
                if direct is not None:
                    y_seg = direct[start:end]
                    y_seg = (y_seg - y_seg.mean()) / (y_seg.std() + 1e-8)
                    y = y_seg.astype(np.float32)
                samples.append(
                    WindowSample(x=x, y=y, p=p_soft, channel_mask=mask, meta={"record": hea.stem, "start": start})
                )
        logger.info("ADFECGDBDataset: %d windows from %s", len(samples), root)
        super().__init__(samples)
