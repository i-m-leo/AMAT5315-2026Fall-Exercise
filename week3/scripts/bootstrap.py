#!/usr/bin/env python3
"""Block-bootstrap finite-size Ising susceptibility peak fits."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week3/artifacts"
OUTPUT = ROOT / "week3/evidence/chi-bootstrap.png"
BLOCK_LENGTHS = (2000, 4000, 8000)
REPLICATES = 500
SEED = 2026
TC = 2.26919


def read_window(size):
    name = f"window-l{size}"
    run_dir = ARTIFACTS / name
    rows = [json.loads(line) for line in (run_dir / "series.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["T"]].append(abs(row["M"]))
    return {temperature: np.asarray(values, dtype=float) for temperature, values in grouped.items()}


def circular_block_sum(values, starts, block_length):
    n = len(values)
    doubled = np.concatenate((values, values))
    cumulative = np.concatenate(([0.0], np.cumsum(doubled)))
    ends = starts + block_length
    return cumulative[ends] - cumulative[starts]


def bootstrap_fit(values_by_temperature, size, block_length, rng):
    temperatures = np.array(sorted(values_by_temperature))
    fitted = []
    for _ in range(REPLICATES):
        chi = []
        for temperature in temperatures:
            values = values_by_temperature[temperature]
            n_blocks = int(np.ceil(len(values) / block_length))
            starts = rng.integers(0, len(values), size=n_blocks)
            sums_abs = circular_block_sum(values, starts, block_length)
            sums_sq = circular_block_sum(values * values, starts, block_length)
            total = n_blocks * block_length
            mean_abs = sums_abs.sum() / total
            mean_sq = sums_sq.sum() / total
            chi.append(size * size * (mean_sq - mean_abs * mean_abs) / temperature)
        chi = np.asarray(chi)
        peak = int(np.argmax(chi))
        lo, hi = peak - 2, peak + 3
        if lo < 0 or hi > len(temperatures):
            raise RuntimeError(f"peak for L={size} is too close to the grid boundary")
        coefficients = np.polyfit(temperatures[lo:hi], chi[lo:hi], 2)
        fitted.append((coefficients, peak))
    return temperatures, fitted


def main():
    data = {size: read_window(size) for size in (32, 64)}
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {32: "tab:blue", 64: "tab:orange"}
    for axis, block_length in zip(axes, BLOCK_LENGTHS):
        for size in (32, 64):
            temperatures, fits = bootstrap_fit(data[size], size, block_length, rng)
            original = []
            for temperature in temperatures:
                values = data[size][temperature]
                mean_abs = np.mean(values)
                original.append(size * size * (np.mean(values * values) - mean_abs * mean_abs) / temperature)
            original = np.asarray(original)
            axis.plot(temperatures, original, "o", ms=2.5, color=colors[size], alpha=0.45)
            fit_grid = np.linspace(2.20, 2.40, 200)
            curves = np.asarray([np.polyval(coefficients, fit_grid) for coefficients, _ in fits])
            axis.fill_between(fit_grid, curves.min(axis=0), curves.max(axis=0), color=colors[size], alpha=0.18)
            axis.plot(fit_grid, curves.mean(axis=0), color=colors[size], label=f"L = {size}")
        axis.axvline(TC, color="black", linestyle="--", linewidth=1, label="Onsager Tc" if block_length == BLOCK_LENGTHS[0] else None)
        axis.set_title(f"block length = {block_length}")
        axis.set_xlabel("temperature T")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("susceptibility χ(T)")
    axes[0].legend()
    fig.suptitle("Block-bootstrap susceptibility peak fits (500 replicates)")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
