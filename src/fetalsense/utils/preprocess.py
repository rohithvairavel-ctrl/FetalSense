"""Shared ECG preprocessing utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    x: np.ndarray,
    fs: float = 250.0,
    low: float = 3.0,
    high: float = 90.0,
    order: int = 4,
    notch_hz: Optional[float] = None,
) -> np.ndarray:
    """Bandpass (Butterworth) with optional notch. x shape (C, T) or (T,)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    nyq = 0.5 * fs
    high_c = min(high, nyq * 0.99)
    low_c = max(low, 0.01)
    b, a = butter(order, [low_c / nyq, high_c / nyq], btype="band")
    y = np.stack([filtfilt(b, a, ch) for ch in x], axis=0)
    if notch_hz is not None and 0 < notch_hz < nyq:
        bn, an = iirnotch(notch_hz, Q=30.0, fs=fs)
        y = np.stack([filtfilt(bn, an, ch) for ch in y], axis=0)
    return y[0] if squeeze else y


def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score over time. x: (C, T) or (B, C, T)."""
    x = np.asarray(x, dtype=np.float64)
    axis = -1
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def make_soft_qrs_targets(
    peak_indices: np.ndarray,
    length: int,
    fs: float = 250.0,
    sigma_ms: float = 25.0,
) -> np.ndarray:
    """Gaussian bumps centered on reference R-peaks → soft targets in [0, 1]."""
    target = np.zeros(length, dtype=np.float64)
    sigma = max(1.0, sigma_ms * 1e-3 * fs)
    t = np.arange(length, dtype=np.float64)
    for pk in np.asarray(peak_indices, dtype=np.int64).ravel():
        if 0 <= pk < length:
            bump = np.exp(-0.5 * ((t - pk) / sigma) ** 2)
            target = np.maximum(target, bump)
    return target
