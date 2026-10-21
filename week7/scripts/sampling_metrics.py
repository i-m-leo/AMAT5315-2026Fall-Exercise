"""Numerical metrics used for sampler comparisons."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def energy_distance_unbiased(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Unbiased U-statistic estimate of the Euclidean energy distance."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("both sample arrays must have shape (n, d) with matching d")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("energy distance requires at least two points in each set")
    between = cdist(a, b, metric="euclidean").mean()
    within_a = pdist(a, metric="euclidean").mean()
    within_b = pdist(b, metric="euclidean").mean()
    return float(2.0 * between - within_a - within_b)


def relative_rms_error(predicted: np.ndarray, exact: np.ndarray) -> float:
    """RMS prediction error divided by RMS magnitude of the exact field."""
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if predicted.shape != exact.shape or predicted.size == 0:
        raise ValueError("predicted and exact arrays must be non-empty with equal shapes")
    denominator = np.sqrt(np.mean(np.square(exact)))
    if denominator == 0:
        raise ValueError("exact field has zero RMS")
    return float(np.sqrt(np.mean(np.square(predicted - exact))) / denominator)
