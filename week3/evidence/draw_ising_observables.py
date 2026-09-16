#!/usr/bin/env python3
"""Draw Ising magnetization and susceptibility evidence plots.

Run with:
MPLCONFIGDIR=/private/tmp/ising-mplconfig /Users/leo/anaconda3/envs/ising-plot/bin/python week3/evidence/draw_ising_observables.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week3/evidence"


def read_run(name):
    run_path = ROOT / "week3/artifacts" / name / "run.json"
    data = json.loads(run_path.read_text())
    rows = [json.loads(line) for line in (run_path.parent / "series.jsonl").read_text().splitlines()]
    by_temperature = {}
    for row in rows:
        by_temperature.setdefault(row["T"], []).append(row)
    return data["L"], by_temperature


def averages(by_temperature):
    result = {}
    for temperature, rows in by_temperature.items():
        m = [row["M"] for row in rows]
        result[temperature] = (sum(abs(value) for value in m) / len(m), sum(value * value for value in m) / len(m))
    return result


def merged(coarse, window):
    result = dict(coarse)
    result.update(window)
    return result


def onsager(temperature):
    critical = 2.0 / math.log(1.0 + math.sqrt(2.0))
    if temperature >= critical:
        return 0.0
    return (1.0 - math.sinh(2.0 / temperature) ** -4) ** 0.125


def main():
    l32c, l32c_rows = read_run("coarse-l32")
    l64c, l64c_rows = read_run("coarse-l64")
    l32w, l32w_rows = read_run("window-l32")
    l64w, l64w_rows = read_run("window-l64")
    assert (l32c, l64c, l32w, l64w) == (32, 64, 32, 64)
    coarse32, coarse64 = averages(l32c_rows), averages(l64c_rows)
    window32, window64 = averages(l32w_rows), averages(l64w_rows)
    susceptibility32 = merged(coarse32, window32)
    susceptibility64 = merged(coarse64, window64)

    temperatures = sorted(coarse64)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temperatures, [coarse64[t][0] for t in temperatures], "o-", label="Metropolis, L = 64")
    theory_t = [1.5 + 0.01 * i for i in range(201)]
    ax.plot(theory_t, [onsager(t) for t in theory_t], "k--", label="Onsager, infinite lattice")
    ax.set(xlabel="temperature T", ylabel="mean |M|", title="Ising magnetization")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "magnetization.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for l, values in ((32, susceptibility32), (64, susceptibility64)):
        ts = sorted(values)
        chi = [l * l * (values[t][1] - values[t][0] ** 2) / t for t in ts]
        ax.plot(ts, chi, "o-", ms=3, label=f"Metropolis, L = {l}")
    ax.axvline(2.0 / math.log(1.0 + math.sqrt(2.0)), color="0.35", linestyle=":", label="Onsager Tc")
    ax.set(xlabel="temperature T", ylabel="susceptibility χ(T)", title="Ising susceptibility")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVIDENCE / "susceptibility.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
