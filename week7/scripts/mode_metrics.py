"""Nearest-centre assignments and empirical Gaussian-mixture summaries."""
from __future__ import annotations

import numpy as np


def assign_nearest_mode(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each 2-D sample to its nearest reference mixture centre."""
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if samples.ndim != 2 or centers.ndim != 2 or samples.shape[1] != centers.shape[1]:
        raise ValueError("samples and centers must be 2-D arrays with matching coordinate dimensions")
    return np.square(samples[:, None, :] - centers[None, :, :]).sum(axis=-1).argmin(axis=1)


def summarize_modes(samples: np.ndarray, centers: np.ndarray,
                    labels: np.ndarray | None = None) -> list[dict]:
    """Measure count, fraction, sample centre, variance, and SD per mode.

    If labels are omitted, samples are assigned to the nearest supplied centre.
    Variance and standard deviation use population convention (ddof=0).
    """
    samples = np.asarray(samples, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if samples.ndim != 2 or centers.ndim != 2 or samples.shape[1] != centers.shape[1]:
        raise ValueError("samples and centers must be 2-D arrays with matching coordinate dimensions")
    if len(samples) == 0 or len(centers) == 0:
        raise ValueError("samples and centers must be non-empty")
    if labels is None:
        labels = assign_nearest_mode(samples, centers)
    else:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.shape != (len(samples),) or np.any(labels < 0) or np.any(labels >= len(centers)):
            raise ValueError("labels must contain one valid mode index per sample")

    results = []
    for mode_index, reference_center in enumerate(centers):
        subset = samples[labels == mode_index]
        if len(subset):
            mean = subset.mean(axis=0)
            variance = subset.var(axis=0, ddof=0)
            std = np.sqrt(variance)
            empirical_center = mean.tolist()
            coordinate_variance = variance.tolist()
            coordinate_std = std.tolist()
        else:
            empirical_center = [None] * samples.shape[1]
            coordinate_variance = [None] * samples.shape[1]
            coordinate_std = [None] * samples.shape[1]
        population = len(subset)
        results.append({
            "mode_index": mode_index,
            "reference_center": reference_center.tolist(),
            "population_count": population,
            "population_fraction": population / len(samples),
            "empirical_center": empirical_center,
            "coordinate_variance": coordinate_variance,
            "coordinate_std": coordinate_std,
        })
    return results
