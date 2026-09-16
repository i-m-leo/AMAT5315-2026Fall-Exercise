#!/usr/bin/env python3
"""Fit finite-size susceptibility peaks for the 2D Ising runs."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week3/artifacts"
OUTPUT = ROOT / "week3/evidence/peaks.txt"


def read_rows(name):
    run = json.loads((ARTIFACTS / name / "run.json").read_text())
    rows = [json.loads(line) for line in (ARTIFACTS / name / "series.jsonl").read_text().splitlines()]
    return run["L"], rows


def susceptibility(name):
    size, rows = read_rows(name)
    temperatures = sorted({row["T"] for row in rows})
    values = []
    for temperature in temperatures:
        magnetizations = np.array([row["M"] for row in rows if row["T"] == temperature])
        mean_abs = np.mean(np.abs(magnetizations))
        values.append(size * size * (np.mean(magnetizations ** 2) - mean_abs ** 2) / temperature)
    return size, np.array(temperatures), np.array(values)


def fitted_peak(name):
    size, temperatures, values = susceptibility(name)
    peak_index = int(np.argmax(values))
    if peak_index < 2 or peak_index + 2 >= len(temperatures):
        raise ValueError(f"largest peak for L={size} is too close to the temperature-grid boundary")
    window = slice(peak_index - 2, peak_index + 3)
    coefficients = np.polyfit(temperatures[window], values[window], 2)
    a, b, c = coefficients
    t_peak = -b / (2 * a)
    chi_peak = np.polyval(coefficients, t_peak)
    return size, t_peak, chi_peak


def cold_mean(name):
    size, rows = read_rows(name)
    cold = [abs(row["M"]) for row in rows if row["T"] == 1.5]
    if not cold:
        raise ValueError(f"no T=1.5 rows for L={size}")
    return size, float(np.mean(cold))


def main():
    peak32 = fitted_peak("window-l32")
    peak64 = fitted_peak("window-l64")
    cold32 = cold_mean("coarse-l32")
    cold64 = cold_mean("coarse-l64")
    tc = 2.0 * peak64[1] - peak32[1]
    text = (
        f"L=32 cold mean |M| = {cold32[1]:.4f}\n"
        f"L=64 cold mean |M| = {cold64[1]:.4f}\n"
        f"L=32 fitted T_peak = {peak32[1]:.4f} (chi_peak = {peak32[2]:.4f})\n"
        f"L=64 fitted T_peak = {peak64[1]:.4f} (chi_peak = {peak64[2]:.4f})\n"
        f"T_c = 2*T_peak(64) - T_peak(32) = {tc:.4f}\n"
    )
    print(text, end="")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)


if __name__ == "__main__":
    main()
