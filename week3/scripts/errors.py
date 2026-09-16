#!/usr/bin/env python3
"""Estimate sampling errors and integrated autocorrelation times for |M|."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week3/artifacts"
OUTPUT = ROOT / "week3/evidence/errors.txt"


def series(name):
    rows = [json.loads(line) for line in (ARTIFACTS / name / "series.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["T"]].append(abs(row["M"]))
    return json.loads((ARTIFACTS / name / "run.json").read_text())["L"], grouped


def autocorrelation_time(values):
    x = np.asarray(values, dtype=float)
    x -= x.mean()
    variance = np.dot(x, x) / len(x)
    if variance == 0:
        return 0.5
    size = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    correlation = np.fft.irfft(spectrum * spectrum.conjugate(), size)[: len(x)]
    correlation /= np.arange(len(x), 0, -1)
    rho = correlation / correlation[0]
    tau = 0.5
    lag = 1
    while lag < len(rho) and lag <= 6.0 * tau:
        tau += rho[lag]
        lag += 1
    return max(0.5, float(tau))


def estimates(values):
    x = np.asarray(values, dtype=float)
    n = len(x)
    naive = float(np.std(x, ddof=1) / np.sqrt(n))
    block_count = 50
    block_size = n // block_count
    blocks = x[: block_count * block_size].reshape(block_count, block_size).mean(axis=1)
    block_error = float(np.std(blocks, ddof=1) / np.sqrt(block_count))
    return float(x.mean()), naive, block_error, block_error / naive if naive else float("nan"), autocorrelation_time(x)


def main():
    runs = {32: series("coarse-l32"), 64: series("coarse-l64")}
    windows = {32: series("window-l32"), 64: series("window-l64")}
    lines = ["L\tT\tmean_abs_M\tnaive_se\tblock50_se\tratio\ttau_int"]
    for size in (32, 64):
        _, coarse = runs[size]
        _, window = windows[size]
        grouped = dict(coarse)
        grouped.update(window)
        for temperature in sorted(grouped):
            mean, naive, block, ratio, tau = estimates(grouped[temperature])
            lines.append(f"{size}\t{temperature:.2f}\t{mean:.6f}\t{naive:.6f}\t{block:.6f}\t{ratio:.3f}\t{tau:.3f}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)


if __name__ == "__main__":
    main()
