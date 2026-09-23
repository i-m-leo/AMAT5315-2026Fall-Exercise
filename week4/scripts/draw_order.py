"""Plot the time-step error of RK4 on Taylor-Green, on log-log axes.

Reads the runs written by scripts/run_order.sh, computes the relative L2 error
of each final velocity field against the exact solution at t = 2, fits a
straight line in log-log space, and labels the slope in the legend.

Writes evidence/order.png. Run from week4/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ORDER = ROOT / "artifacts" / "order"
OUT = ROOT / "evidence" / "order.png"

DTS = [0.4, 0.25, 0.2]


def last_frame(path):
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return frame


def relative_error(frame, exact):
    u = np.asarray(frame["u"], dtype=float)
    v = np.asarray(frame["v"], dtype=float)
    ue = np.asarray(exact["u"], dtype=float)
    ve = np.asarray(exact["v"], dtype=float)
    numerator = np.sum((u - ue) ** 2 + (v - ve) ** 2)
    denominator = np.sum(ue**2 + ve**2)
    return np.sqrt(numerator / denominator)


def main():
    with (ORDER / "exact-t2.json").open() as handle:
        exact = json.load(handle)

    dts, errors = [], []
    print("Taylor-Green, n = 8, nu = 0.5, to t = 2, RK4")
    print(f"{'dt':>6} {'steps':>6} {'relative error':>18}")
    for dt in DTS:
        frame = last_frame(ORDER / f"rk4-dt{dt}" / "fields-full.jsonl")
        error = relative_error(frame, exact)
        dts.append(dt)
        errors.append(error)
        print(f"{dt:>6} {round(2.0 / dt):>6} {error:>18.6e}")
    dts = np.asarray(dts)
    errors = np.asarray(errors)

    slope, intercept = np.polyfit(np.log(dts), np.log(errors), 1)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.loglog(dts, errors, "o", color="#1f77b4", markersize=8,
              label="RK4")
    fit_x = np.linspace(dts.min(), dts.max(), 50)
    ax.loglog(fit_x, np.exp(intercept) * fit_x**slope, "-", color="#1f77b4",
              linewidth=1.2, label=f"fit, slope {slope:.2f}")

    for dt, error in zip(dts, errors):
        ax.annotate(f"{error:.2e}", xy=(dt, error), xytext=(6, 6),
                    textcoords="offset points", fontsize=8)

    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("relative velocity error at $t = 2$")
    ax.set_title("RK4 order on Taylor-Green, n = 8, $\\nu$ = 0.5")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.savefig(OUT, dpi=140)
    print(f"fitted slope = {slope:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
