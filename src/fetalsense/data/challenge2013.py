"""PhysioNet Computing in Cardiology Challenge 2013 (Set A) loader sketch.

Expected layout under ``paths.challenge2013``::

    a01.dat  a01.hea  a01.fqrs
    a02.dat  ...

Uses ``wfdb`` when installed and records are present; otherwise returns an empty dataset.
Does not fail import or tests when data is missing.
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
N_CH = 4


def _load_record(record_path: Path, target_fs: int = TARGET_FS) -> Optional[tuple]:
    try:
        import wfdb
    except ImportError:
        logger.info("wfdb not installed; Challenge-2013 loader inactive.")
        return None
    try:
        rec = wfdb.rdrecord(str(record_path.with_suffix("")))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to read %s: %s", record_path, exc)
        return None
    sig = np.asarray(rec.p_signal, dtype=np.float64).T  # (C, T)
    fs = float(rec.fs)
    # Resample if needed
    if abs(fs - target_fs) > 1e-3:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(target_fs), int(fs))
        up, down = int(target_fs) // g, int(fs) // g
        sig = np.stack([resample_poly(ch, up, down) for ch in sig], axis=0)
        fs = float(target_fs)

    # Optional annotation file *.fqrs or *.qrs
    peaks: List[int] = []
    for ext in (".fqrs", ".qrs", ".atr"):
        ann_path = record_path.with_suffix(ext)
        if ann_path.exists():
            try:
                ann = wfdb.rdann(str(record_path.with_suffix("")), extension=ext.lstrip("."))
                peaks = [int(round(s * target_fs / fs)) if abs(fs - target_fs) > 1e-3 else int(s) for s in ann.sample]
            except Exception:  # noqa: BLE001
                peaks = []
            break
    return sig, peaks, fs


class Challenge2013Dataset(FetalECGWindowDataset):
    """Sliding-window dataset over Challenge 2013 records (if present)."""

    def __init__(
        self,
        root: str | Path,
        window: int = WINDOW,
        stride: int = 256,
        n_channels: int = N_CH,
        max_records: Optional[int] = None,
    ) -> None:
        root = Path(root)
        samples: List[WindowSample] = []
        if not root.exists():
            logger.warning("Challenge-2013 path missing: %s (empty dataset)", root)
            super().__init__(samples)
            return

        hea_files = sorted(root.glob("*.hea"))
        if max_records is not None:
            hea_files = hea_files[:max_records]

        for hea in hea_files:
            loaded = _load_record(hea)
            if loaded is None:
                continue
            sig, peaks, _fs = loaded
            sig = bandpass_filter(sig, fs=TARGET_FS)
            # Pad / crop channels
            C = sig.shape[0]
            if C < n_channels:
                pad = np.zeros((n_channels - C, sig.shape[1]), dtype=sig.dtype)
                mask = np.array([1] * C + [0] * (n_channels - C), dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=0)
            else:
                sig = sig[:n_channels]
                mask = np.ones(n_channels, dtype=np.float32)

            T = sig.shape[1]
            for start in range(0, max(1, T - window + 1), stride):
                end = start + window
                if end > T:
                    break
                x = zscore_per_channel(sig[:, start:end])
                local_peaks = [p - start for p in peaks if start <= p < end]
                p_soft = make_soft_qrs_targets(np.array(local_peaks), window, fs=TARGET_FS)
                samples.append(
                    WindowSample(
                        x=x.astype(np.float32),
                        p=p_soft.astype(np.float32),
                        channel_mask=mask,
                        meta={"record": hea.stem, "start": start},
                    )
                )
        logger.info("Challenge2013Dataset: %d windows from %s", len(samples), root)
        super().__init__(samples)
