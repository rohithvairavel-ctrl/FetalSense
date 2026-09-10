"""Unit tests for peak picking and fQRS metrics."""

from __future__ import annotations

import numpy as np

from fetalsense.metrics.fqrs import match_peaks, score_probability_map, sens_ppv_f1
from fetalsense.utils.peaks import pick_peaks
from fetalsense.utils.preprocess import make_soft_qrs_targets


def test_pick_peaks_refractory():
    T = 512
    fs = 250.0
    # Place two strong peaks 100 ms apart (should keep only one with 180 ms refractory)
    p = np.zeros(T)
    p[100] = 0.9
    p[125] = 0.85  # 100 ms later @ 250 Hz = 25 samples
    peaks = pick_peaks(p, fs=fs, threshold=0.3, refractory_ms=180)
    assert len(peaks) == 1
    assert peaks[0] == 100

    # Far peaks both kept
    p2 = np.zeros(T)
    p2[50] = 0.9
    p2[200] = 0.9  # 600 ms apart
    peaks2 = pick_peaks(p2, fs=fs, threshold=0.3, refractory_ms=180)
    assert list(peaks2) == [50, 200]


def test_sens_ppv_f1_perfect():
    ref = np.array([10, 50, 100])
    det = np.array([10, 50, 100])
    m = sens_ppv_f1(ref, det, fs=250.0, tolerance_ms=50.0)
    assert m["f1"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["ppv"] == 1.0


def test_match_tolerance_50_vs_100():
    ref = np.array([100])
    det = np.array([120])  # 80 ms @ 250 Hz (20 samples)
    # 50 ms = 12.5 → 12 samples → miss; 100 ms = 25 samples → hit
    tp50, _, _ = match_peaks(ref, det, tolerance_samples=12)
    tp100, _, _ = match_peaks(ref, det, tolerance_samples=25)
    assert tp50 == 0
    assert tp100 == 1


def test_score_probability_map():
    ref = np.array([64, 200, 350])
    soft = make_soft_qrs_targets(ref, length=512, fs=250.0, sigma_ms=25.0)
    scores = score_probability_map(soft, ref, fs=250.0, tolerances_ms=(50, 100), threshold=0.3)
    assert "50ms" in scores and "100ms" in scores
    assert scores["50ms"]["f1"] >= 0.9
