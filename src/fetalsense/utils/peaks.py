"""Refractory peak picking for fetal QRS likelihood maps."""

from __future__ import annotations

import numpy as np


def pick_peaks(
    prob: np.ndarray,
    fs: float = 250.0,
    threshold: float = 0.3,
    refractory_ms: float = 180.0,
) -> np.ndarray:
    """Greedy local-maxima peak pick with refractory period.

    Args:
        prob: 1-D array of QRS likelihood in [0, 1], length T.
        fs: Sample rate in Hz.
        threshold: Minimum amplitude to consider a peak.
        refractory_ms: Minimum spacing between accepted peaks (ms).

    Returns:
        Integer sample indices of accepted peaks (ascending).
    """
    x = np.asarray(prob, dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.int64)

    refractory = max(1, int(round(refractory_ms * 1e-3 * fs)))
    # Candidate local maxima above threshold
    candidates: list[int] = []
    for i in range(1, len(x) - 1):
        if x[i] >= threshold and x[i] >= x[i - 1] and x[i] > x[i + 1]:
            candidates.append(i)
    # Edge cases
    if len(x) >= 1 and x[0] >= threshold and (len(x) == 1 or x[0] > x[1]):
        candidates.insert(0, 0)
    if len(x) >= 2 and x[-1] >= threshold and x[-1] > x[-2]:
        candidates.append(len(x) - 1)

    if not candidates:
        return np.array([], dtype=np.int64)

    # Sort by amplitude descending, then enforce refractory in time order of acceptance
    order = sorted(candidates, key=lambda i: -x[i])
    accepted: list[int] = []
    for idx in order:
        if all(abs(idx - a) >= refractory for a in accepted):
            accepted.append(idx)
    return np.array(sorted(accepted), dtype=np.int64)
