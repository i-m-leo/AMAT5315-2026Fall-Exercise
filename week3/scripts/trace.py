#!/usr/bin/env python3
"""Draw the first 2000 absolute-magnetization sweeps at T=2.3 and 3.0."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SERIES = ROOT / "week3/artifacts/coarse-l64/series.jsonl"
OUTPUT = ROOT / "week3/evidence/trace.png"


def main():
    traces = defaultdict(list)
    for line in SERIES.read_text().splitlines():
        row = json.loads(line)
        if row["T"] in (2.3, 3.0) and row["sweep"] <= 2000:
            traces[row["T"]].append((row["sweep"], abs(row["M"])))
    if any(len(traces[t]) != 2000 for t in (2.3, 3.0)):
        raise RuntimeError("expected 2000 recorded sweeps at both temperatures")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for temperature, color in ((2.3, "tab:blue"), (3.0, "tab:orange")):
        sweeps, magnetization = zip(*traces[temperature])
        ax.plot(sweeps, magnetization, linewidth=0.8, color=color, label=f"T = {temperature}")
    ax.set(xlabel="recorded sweep", ylabel="|M|", title="Metropolis magnetization traces, L = 64")
    ax.set_xlim(1, 2000)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
