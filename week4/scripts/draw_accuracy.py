"""Draw the week4 line-accuracy figure.

Top two panels: the final profiles of the one-lap runs against the exact
periodic solution, and their pointwise error (from `accuracy_run`).
Bottom panel: maximum error against time step, on log-log axes, for forward
Euler, explicit midpoint, classical RK4, and an equal-weight RK4, each with a
linear fit in log-log space whose slope is reported in the legend
(from `method_convergence`).

Run from week4/.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
PROFILES = EVIDENCE / "line-accuracy-profiles.txt"
CONVERGENCE = EVIDENCE / "method-convergence.txt"
OUT = EVIDENCE / "line-accuracy.png"

SERIES = [
    ("RK4 Fourier  dt=0.02", "#1f77b4", "-"),
    ("RK4 centred  dt=0.02", "#2ca02c", "--"),
    ("Euler Fourier dt=0.005", "#d62728", "-."),
]

METHODS = [
    ("forward-euler", "forward Euler", "#d62728", "o"),
    ("explicit-midpoint", "explicit midpoint", "#2ca02c", "s"),
    ("classical-rk4", "classical RK4", "#1f77b4", "^"),
    ("rk4-equal-weights", "RK4 equal weights", "#9467bd", "D"),
]


def draw_profiles(ax, ax_err):
    with PROFILES.open() as handle:
        header = handle.readline().split()
        values = np.loadtxt(handle)
    x = values[:, 0]
    exact = values[:, 1]
    profiles = values[:, 2:]
    assert len(SERIES) == profiles.shape[1], (len(SERIES), header)

    ax.plot(x, exact, color="black", linewidth=2.0, label="exact")
    for column, (label, color, style) in zip(profiles.T, SERIES):
        ax.plot(x, column, color=color, linestyle=style, linewidth=1.3,
                label=label)
    ax.set_ylabel("u(x, 2$\\pi$)")
    ax.set_title("Periodic Gaussian pulse after one lap, n = 64, "
                 "c = 1, $\\nu$ = 0.002")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 2.0 * np.pi)

    for column, (label, color, style) in zip(profiles.T, SERIES):
        ax_err.plot(x, column - exact, color=color, linestyle=style,
                    linewidth=1.2, label=label)
    ax_err.axhline(0.0, color="black", linewidth=0.6)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("u - exact")
    ax_err.grid(alpha=0.25)
    ax_err.set_xlim(0.0, 2.0 * np.pi)


def draw_convergence(ax):
    table = np.genfromtxt(CONVERGENCE, names=True, dtype=None, encoding="utf-8")

    for key, label, color, marker in METHODS:
        rows = table[table["method"] == key]
        dt = rows["dt"]
        error = rows["max_error"]
        ax.loglog(dt, error, marker=marker, color=color, linestyle="none",
                  markersize=6, label=label)

        slope, intercept = np.polyfit(np.log(dt), np.log(error), 1)
        fit_x = np.array([dt.min(), dt.max()])
        ax.loglog(fit_x, np.exp(intercept) * fit_x**slope, color=color,
                  linestyle="-", linewidth=1.0, alpha=0.7)
        ax.plot([], [], color=color, linestyle="-", linewidth=1.0,
                label=f"{label}  slope {slope:.2f}")

    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("max $|u - u_{exact}|$")
    ax.set_title("Convergence: Gaussian pulse $\\sigma$ = 0.35, n = 64, "
                 "c = 1, $\\nu$ = 0.05, t = 1 (Fourier derivatives)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)


def main():
    fig = plt.figure(figsize=(10.5, 11.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 2.0])

    ax_profile = fig.add_subplot(grid[0])
    ax_err = fig.add_subplot(grid[1], sharex=ax_profile)
    ax_conv = fig.add_subplot(grid[2])

    draw_profiles(ax_profile, ax_err)
    draw_convergence(ax_conv)

    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
