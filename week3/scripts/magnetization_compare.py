#!/usr/bin/env python3
"""Compare Metropolis and Wolff magnetization and susceptibility for L=64."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week3/artifacts"
OUTPUT = ROOT / "week3/evidence/magnetization-compare.png"
BLOCK_LENGTH = 2000
BOOTSTRAPS = 500
TC_EXACT = 2.26919


def read_run(name):
    rows = defaultdict(list)
    path = ARTIFACTS / name / "series.jsonl"
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row["T"]].append(abs(row["M"]))
    return {temperature: np.asarray(values, dtype=float) for temperature, values in rows.items()}


def stats(values, rng):
    mean = float(values.mean())
    block_count = len(values) // BLOCK_LENGTH
    block_means = values[: block_count * BLOCK_LENGTH].reshape(block_count, BLOCK_LENGTH).mean(axis=1)
    bootstrap_means = np.array([block_means[rng.integers(0, block_count, block_count)].mean() for _ in range(BOOTSTRAPS)])
    return mean, float(bootstrap_means.std(ddof=1))


def susceptibility(values, temperature, size):
    return size * size * (np.mean(values * values) - np.mean(values) ** 2) / temperature


def peak(values_by_temperature, size):
    temperatures = np.array(sorted(values_by_temperature))
    chi = np.array([susceptibility(values_by_temperature[t], t, size) for t in temperatures])
    index = int(np.argmax(chi))
    coefficients = np.polyfit(temperatures[index - 2:index + 3], chi[index - 2:index + 3], 2)
    t_peak = -coefficients[1] / (2 * coefficients[0])
    return temperatures, chi, coefficients, t_peak


def main():
    metro = {32: read_run("window-l32"), 64: read_run("window-l64")}
    wolff = {32: read_run("wolff-l32"), 64: read_run("wolff-l64")}
    rng = np.random.default_rng(2026)
    fig, (mag_axis, chi_axis) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    colors = {"Metropolis": "tab:blue", "Wolff": "tab:orange"}

    for label, runs in (("Metropolis", metro[64]), ("Wolff", wolff[64])):
        temperatures = np.array(sorted(runs))
        means, errors = zip(*(stats(runs[t], rng) for t in temperatures))
        mag_axis.errorbar(temperatures, means, yerr=errors, fmt="o-", capsize=2, color=colors[label], label=label)
        ts, chi, coefficients, t_peak = peak(runs, 64)
        fit_x = np.linspace(ts[np.argmax(chi) - 2], ts[np.argmax(chi) + 2], 150)
        chi_axis.plot(ts, chi, "o", ms=3, color=colors[label])
        chi_axis.plot(fit_x, np.polyval(coefficients, fit_x), color=colors[label], label=f"{label} five-point fit")

        _, _, _, t32 = peak(metro[32] if label == "Metropolis" else wolff[32], 32)
        tc = 2 * t_peak - t32
        chi_axis.axvline(tc, color=colors[label], linestyle=":", linewidth=1, label=f"{label} extrapolated Tc = {tc:.4f}")

    mag_axis.set(ylabel="mean |M|", title="L = 64 magnetization: Metropolis vs Wolff")
    mag_axis.grid(alpha=0.25)
    mag_axis.legend()
    chi_axis.axvline(TC_EXACT, color="black", linestyle="--", linewidth=1, label="Onsager Tc = 2.26919")
    chi_axis.set(xlabel="temperature T", ylabel="cluster susceptibility χ(T)", title="Susceptibility and five-point peak fits")
    chi_axis.grid(alpha=0.25)
    chi_axis.legend(fontsize=8)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
