"""Plot the three final profiles against the exact periodic solution.

Reads the profiles written by `accuracy_run` and draws
evidence/line-accuracy.png. Run from week4/.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
PROFILES = EVIDENCE / "line-accuracy-profiles.txt"
OUT = EVIDENCE / "line-accuracy.png"


def main():
    with PROFILES.open() as handle:
        header = handle.readline().split()
        values = np.loadtxt(handle)
    x = values[:, 0]
    exact = values[:, 1]
    profiles = values[:, 2:]

    series = [
        ("RK4 Fourier  dt=0.02", "#1f77b4", "-"),
        ("RK4 centred  dt=0.02", "#2ca02c", "--"),
        ("Euler Fourier dt=0.005", "#d62728", "-."),
    ]
    assert len(series) == profiles.shape[1], (len(series), header)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.0, 1.0]})

    ax = axes[0]
    ax.plot(x, exact, color="black", linewidth=2.0, label="exact")
    for column, (label, color, style) in zip(profiles.T, series):
        ax.plot(x, column, color=color, linestyle=style, linewidth=1.3,
                label=label)
    ax.set_ylabel("u(x, 2$\\pi$)")
    ax.set_title("Periodic Gaussian pulse after one lap, n = 64, "
                 "c = 1, $\\nu$ = 0.002")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 2.0 * np.pi)

    ax = axes[1]
    for column, (label, color, style) in zip(profiles.T, series):
        ax.plot(x, column - exact, color=color, linestyle=style,
                linewidth=1.2, label=label)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("u - exact")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.0, 2.0 * np.pi)

    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
