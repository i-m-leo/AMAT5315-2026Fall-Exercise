#!/usr/bin/env python3
"""Compare autocorrelation time at equal work in spin-update sweeps."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week3/artifacts"
OUTPUT = ROOT / "week3/evidence/tau-compare.png"


def read_series(name):
    grouped = defaultdict(list)
    for line in (ARTIFACTS / name / "series.jsonl").read_text().splitlines():
        row = json.loads(line)
        grouped[row["T"]].append(row)
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
    metro = read_series("window-l64")
    wolff = read_series("wolff-l64")
    temperatures = sorted(set(metro) & set(wolff))
    metro_tau = [tau_int([abs(row["M"]) for row in metro[t]]) for t in temperatures]
    wolff_tau = []
    for temperature in temperatures:
        rows = wolff[temperature]
        move_tau = tau_int([abs(row["M"]) for row in rows])
        mean_cluster = np.mean([row["cluster_size"] for row in rows])
        wolff_tau.append(move_tau * mean_cluster / 64**2)

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(temperatures, metro_tau, "o-", label="Metropolis")
    axis.plot(temperatures, wolff_tau, "o-", label="Wolff (work-normalized)")
    axis.set_yscale("log")
    axis.set(xlabel="temperature T", ylabel="integrated autocorrelation time (spin-update sweeps)",
             title="Work-normalized autocorrelation time, L = 64")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
