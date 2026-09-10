"""fQRS detection metrics with temporal matching tolerance."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from fetalsense.utils.peaks import pick_peaks


def match_peaks(
    ref: np.ndarray,
    det: np.ndarray,
    tolerance_samples: int,
) -> Tuple[int, int, int]:
    """Greedy one-to-one match within ±tolerance.

    Returns:
        (TP, FP, FN)
    """
    ref = np.asarray(ref, dtype=np.int64).ravel()
    det = np.asarray(det, dtype=np.int64).ravel()
    if len(ref) == 0 and len(det) == 0:
        return 0, 0, 0
    if len(ref) == 0:
        return 0, len(det), 0
    if len(det) == 0:
        return 0, 0, len(ref)

    ref_used = np.zeros(len(ref), dtype=bool)
    tp = 0
    for d in det:
        # Find closest unused ref within tolerance
        diffs = np.abs(ref - d)
        diffs[ref_used] = tolerance_samples + 1
        j = int(np.argmin(diffs))
        if diffs[j] <= tolerance_samples:
            ref_used[j] = True
            tp += 1
    fp = len(det) - tp
    fn = len(ref) - tp
    return tp, fp, fn


def sens_ppv_f1(
    ref_peaks: np.ndarray,
    det_peaks: np.ndarray,
    fs: float = 250.0,
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """Sensitivity, PPV, and F1 for peak detection at a given tolerance."""
    tol = int(round(tolerance_ms * 1e-3 * fs))
    tp, fp, fn = match_peaks(ref_peaks, det_peaks, tol)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0.0
    return {"sensitivity": sens, "ppv": ppv, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def score_probability_map(
    p_hat: np.ndarray,
    ref_peaks: np.ndarray,
    fs: float = 250.0,
    tolerances_ms: Tuple[float, ...] = (50.0, 100.0),
    threshold: float = 0.3,
    refractory_ms: float = 180.0,
) -> Dict[str, Dict[str, float]]:
    """Peak-pick ``p_hat`` then score at multiple tolerances."""
    det = pick_peaks(p_hat, fs=fs, threshold=threshold, refractory_ms=refractory_ms)
    return {f"{int(t)}ms": sens_ppv_f1(ref_peaks, det, fs=fs, tolerance_ms=t) for t in tolerances_ms}
