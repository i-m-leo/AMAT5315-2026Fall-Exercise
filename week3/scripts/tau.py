#!/usr/bin/env python3
"""Draw integrated autocorrelation time versus temperature for both sizes."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week3/artifacts"
OUTPUT = ROOT / "week3/evidence/tau.png"


def read_series(name):
    rows = [json.loads(line) for line in (ARTIFACTS / name / "series.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["T"]].append(abs(row["M"]))
    return grouped


def tau_int(values):
    x = np.asarray(values, dtype=float) - np.mean(values)
    n = len(x)
    if not np.any(x):
        return 0.5
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    ac = np.fft.irfft(spectrum * spectrum.conjugate(), size)[:n]
    ac /= np.arange(n, 0, -1)
    rho = ac / ac[0]
    tau = 0.5
    lag = 1
    while lag < n and lag <= 6 * tau:
        tau += rho[lag]
        lag += 1
    return max(0.5, float(tau))


def main():
    fig, axis = plt.subplots(figsize=(10, 6))
    for size, color in ((32, "tab:blue"), (64, "tab:orange")):
        grouped = dict(read_series(f"coarse-l{size}"))
        grouped.update(read_series(f"window-l{size}"))
        temperatures = sorted(grouped)
        taus = [tau_int(grouped[t]) for t in temperatures]
        axis.plot(temperatures, taus, "o-", color=color, label=f"L = {size}")
    axis.set_yscale("log")
    axis.set(xlabel="temperature T", ylabel="integrated autocorrelation time (sweeps)", title="Ising integrated autocorrelation time")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
