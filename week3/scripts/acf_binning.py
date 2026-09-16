#!/usr/bin/env python3
"""Draw autocorrelation and block-binned error diagnostics for L=64, T=2.3."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SERIES = ROOT / "week3/artifacts/window-l64/series.jsonl"
OUTPUT = ROOT / "week3/evidence/acf-binning.png"
BLOCK_LENGTHS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])


def load_values():
    return np.array([
        abs(row["M"])
        for row in (json.loads(line) for line in SERIES.read_text().splitlines())
        if row["T"] == 2.3
    ], dtype=float)


def autocorrelation(values):
    centered = values - values.mean()
    n = len(centered)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, size)
    correlation = np.fft.irfft(spectrum * spectrum.conjugate(), size)[:n]
    correlation /= np.arange(n, 0, -1)
    return correlation / correlation[0]


def block_errors(values):
    errors = []
    for length in BLOCK_LENGTHS:
        blocks = values[: len(values) // length * length].reshape(-1, length).mean(axis=1)
        errors.append(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))
    return np.array(errors)


def main():
    values = load_values()
    acf = autocorrelation(values)
    errors = block_errors(values)
    fig, (acf_axis, error_axis) = plt.subplots(1, 2, figsize=(12, 5))
    lags = np.arange(min(5000, len(acf)))
    acf_axis.plot(lags, acf[: len(lags)], color="tab:blue", linewidth=0.8)
    acf_axis.axhline(0, color="0.4", linewidth=0.8)
    acf_axis.set(xlabel="lag (sweeps)", ylabel="autocorrelation of |M|", title="L = 64, T = 2.3")
    acf_axis.grid(alpha=0.25)
    error_axis.plot(BLOCK_LENGTHS, errors, "o-", color="tab:orange")
    error_axis.set_xscale("log")
    error_axis.set(xlabel="block length (sweeps)", ylabel="standard error of mean |M|", title="Block-binned error")
    error_axis.grid(alpha=0.25, which="both")
    error_axis.annotate("sampling error unresolved", xy=(5000, errors[-1]), xytext=(700, errors[-1] * 0.72), arrowprops={"arrowstyle": "->"})
    fig.suptitle("Autocorrelation and binning diagnostics")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
