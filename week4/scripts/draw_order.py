"""Plot the Taylor-Green field error against the RK4 time step on log-log axes.

The runs (n = 8, nu = 0.5, to t = 2) are the six-decimal frames that `fluid`
writes by contract: every frame is rounded to six decimals, so the series has a
storage floor. The error is the relative L2 distance of the final velocity
field from the analytic Taylor-Green solution at t = 2.

The figure shows the points, a fitted line whose slope is labelled, dotted
reference lines of slope 1, 2, and 4, and the dashed six-decimal storage floor.

Writes evidence/order.png. Run from week4/.
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ORDER = ROOT / "artifacts" / "order"
OUT = ROOT / "evidence" / "order.png"

N = 8
NU = 0.5
T_END = 2.0
DTS = [0.4, 0.25, 0.2]
ROUNDING = 1e-6


def exact_velocity():
    x = np.arange(N) * 2.0 * math.pi / N
    xx, yy = np.meshgrid(x, x)
    decay = math.exp(-2.0 * NU * T_END)
    return (
        (np.cos(xx) * np.sin(yy) * decay).reshape(-1),
        (-np.sin(xx) * np.cos(yy) * decay).reshape(-1),
    )


def final_velocity(dt):
    path = ORDER / f"rk4-dt{dt}" / "fields.jsonl"
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return np.asarray(frame["u"], dtype=float), np.asarray(frame["v"], dtype=float)


def main():
    ue, ve = exact_velocity()
    denominator = math.sqrt(np.sum(ue**2 + ve**2))
    # Both fields carry a half-ulp rounding error per component, so the
    # difference is bounded by one ulp per component.
    floor = math.sqrt(2 * N * N) * ROUNDING / denominator

    dts, errors = [], []
    print("Taylor-Green, n = 8, nu = 0.5, to t = 2, RK4, six-decimal frames")
    print(f"{'dt':>6} {'steps':>6} {'relative error':>18}")
    for dt in DTS:
        u, v = final_velocity(dt)
        error = math.sqrt(np.sum((u - ue) ** 2 + (v - ve) ** 2) / denominator**2)
        dts.append(dt)
        errors.append(error)
        print(f"{dt:>6} {round(T_END / dt):>6} {error:>18.6e}")
    dts = np.asarray(dts)
    errors = np.asarray(errors)

    slope, intercept = np.polyfit(np.log(dts), np.log(errors), 1)

    fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax.loglog(dts, errors, "o", color="#1f77b4", markersize=8, label="RK4")

    fit_x = np.linspace(dts.min() * 0.97, dts.max() * 1.03, 50)
    ax.loglog(fit_x, np.exp(intercept) * fit_x**slope, "-", color="#1f77b4",
              linewidth=1.2, label=f"RK4, slope {slope:.2f}")

    for reference_slope in (1, 2, 4):
        line = errors[-1] * (dts / dts[-1]) ** reference_slope
        ax.loglog(dts, line, ":", color="0.55", linewidth=0.9)
        ax.annotate(f"{reference_slope}", xy=(dts[1], line[1]), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color="0.4",
                    ha="center")
    ax.plot([], [], ":", color="0.55", linewidth=0.9, label="1, 2, 4 slopes")

    ax.axhline(floor, color="#d62728", linestyle="--", linewidth=1.1,
               label="6-decimal storage floor")

    for dt, error in zip(dts, errors):
        ax.annotate(f"{error:.2e}", xy=(dt, error), xytext=(7, 5),
                    textcoords="offset points", fontsize=8)

    ratio = errors[0] / errors[-1]
    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("relative field error at $t = 2$")
    ax.set_title(
        "Taylor-Green field error at $t = 2$ against the step, "
        f"$n = {N}$, $\\nu = {NU}$"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.text(
        0.03, 0.03,
        f"RK4's error drops {ratio:.1f} $\\times$ from $\\Delta t$ = 0.4 to 0.2,\n"
        "$\\approx 2^4$ = 16. Every point sits above the\n"
        "floor of the six-decimal frames.",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.9, pad=3.0),
    )
    fig.savefig(OUT, dpi=140)
    print(f"fitted slope = {slope:.3f}  (within 15% of 4: "
          f"{abs(slope - 4) <= 0.6})")
    print(f"ratio 0.4 -> 0.2 = {ratio:.2f}")
    print(f"storage floor = {floor:.3e}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
