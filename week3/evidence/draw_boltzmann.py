#!/usr/bin/env python3
"""Draw Boltzmann evidence. Command: /Users/leo/anaconda3/envs/ising-plot/bin/python week3/evidence/draw_boltzmann.py"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "week3/evidence/boltzmann.png"
WIDTH = 40


def energies(name):
    path = ROOT / "week3/runs" / name / "series.jsonl"
    return [json.loads(line)["E"] * 4096 for line in path.read_text().splitlines()]


def main():
    e30, e31 = energies("T3.0"), energies("T3.1")
    low = math.floor(min(e30 + e31) / WIDTH) * WIDTH
    high = math.ceil(max(e30 + e31) / WIDTH) * WIDTH
    bins = np.arange(low, high + WIDTH, WIDTH)
    n30, _ = np.histogram(e30, bins=bins)
    n31, _ = np.histogram(e31, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    valid = (n30 >= 5) & (n31 >= 5)
    log_ratio = np.full(n30.shape, np.nan, dtype=float)
    log_ratio[valid] = np.log(n31[valid] / n30[valid])
    slope = 1 / 3.0 - 1 / 3.1
    fig, (hist, ratio) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, height_ratios=(1.2, 1))
    hist.hist(e30, bins=bins, alpha=0.55, label="T = 3.0")
    hist.hist(e31, bins=bins, histtype="step", linewidth=1.8, label="T = 3.1")
    hist.set(title="Total energy histograms and Boltzmann log ratio", ylabel="count")
    hist.legend(); hist.grid(alpha=0.2)
    ratio.plot(centers, log_ratio, "o", ms=3, label="log[count(3.1) / count(3.0)]")
    if valid.any():
        x = centers[valid]
        y = log_ratio[valid]
        ratio.plot(x, y[0] + slope * (x - x[0]), color="green", label=f"slope = {slope:.6f}")
    ratio.axhline(0, color="0.5", linewidth=0.8)
    ratio.set(xlabel="total energy E × 4096", ylabel="log ratio")
    ratio.grid(alpha=0.2); ratio.legend()
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=160)


if __name__ == "__main__":
    main()
