"""Diagnose which data source and quantity reproduce the expected order study.

Compares the dt = 0.2, 0.25, 0.4 Taylor-Green runs (n = 8, nu = 0.5, t = 2)
against the analytic exact solution, using both the 6-decimal fields.jsonl and
the full-precision fields-full.jsonl, and prints the slopes and the 6-decimal
storage floor. Run from week4/.
"""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = ROOT / "artifacts" / "order"

N = 8
NU = 0.5
T_END = 2.0
DTS = [0.4, 0.25, 0.2]


def exact_velocity():
    x = np.arange(N) * 2.0 * math.pi / N
    xx, yy = np.meshgrid(x, x)
    decay = math.exp(-2.0 * NU * T_END)
    return (
        (np.cos(xx) * np.sin(yy) * decay).reshape(-1),
        (-np.sin(xx) * np.cos(yy) * decay).reshape(-1),
    )


def load_last(path):
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return frame


def rel_error(frame, ue, ve):
    u = np.asarray(frame["u"], dtype=float)
    v = np.asarray(frame["v"], dtype=float)
    denom = np.sum(ue**2 + ve**2)
    return math.sqrt(np.sum((u - ue) ** 2 + (v - ve) ** 2) / denom)


def main():
    ue, ve = exact_velocity()
    denom = math.sqrt(np.sum(ue**2 + ve**2))

    print("Taylor-Green, n = 8, nu = 0.5, t = 2; exact velocity norm "
          f"||(u,v)||_2 = {denom:.6f}")
    print("floor if both fields are rounded to 6 decimals: "
          f"{math.sqrt(2 * N * N) * 1e-6 / denom:.6e}")
    print()
    print(f"{'dt':>6} {'6dp error':>16} {'full error':>16} {'ratio 6dp':>12}")
    print(f"{'':>6} {'':>16} {'':>16} {'full error ratio':>12}")

    for name, filename in (("6dp", "fields.jsonl"), ("full", "fields-full.jsonl")):
        dts, errors = [], []
        for dt in DTS:
            frame = load_last(ORDER / f"rk4-dt{dt}" / filename)
            errors.append(rel_error(frame, ue, ve))
            dts.append(dt)
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        print(f"{name}:")
        for dt, err in zip(dts, errors):
            print(f"   dt = {dt:<6} error = {err:.6e}")
        print(f"   slope = {slope:.3f}   ratio 0.4->0.2 = {errors[0]/errors[2]:.2f}")
        print()


if __name__ == "__main__":
    main()
