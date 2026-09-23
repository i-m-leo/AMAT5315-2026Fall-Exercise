"""Draw energy against time for the week4 blow-up scan.

One panel per case. The Taylor-Green panel shows both runs; the random panel
shows the RK4 step that reaches t = 10, an RK4 step that blows up, and the
Euler run. The energy axis is logarithmic, each curve is labelled with its
method and time step, and the stopping time of every unstable run is marked
with a vertical line and an upward arrow to the panel top.

Reads artifacts/scan/*.tsv produced by scripts/run_scan.sh and writes
evidence/blowup.png. Run from week4/.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "artifacts" / "scan"
OUT = ROOT / "evidence" / "blowup.png"

PANELS = [
    (
        "Taylor-Green, n = 64, $\\nu$ = 0.1, to t = 8",
        [
            ("tg-rk4-dt0.032.tsv", "RK4  dt = 0.032", "#1f77b4", "-"),
            ("tg-rk4-dt0.033.tsv", "RK4  dt = 0.033  (blows up)", "#d62728", "--"),
        ],
    ),
    (
        "random, n = 128, $\\nu$ = 0.004, to t = 10",
        [
            ("rand-rk4-dt0.033.tsv", "RK4  dt = 0.033  (reaches t = 10)", "#1f77b4", "-"),
            ("rand-rk4-dt0.038.tsv", "RK4  dt = 0.038  (blows up)", "#d62728", "--"),
            ("rand-euler-dt0.01.tsv", "Euler  dt = 0.01  (blows up)", "#2ca02c", "-."),
        ],
    ),
]


def load(path):
    values = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    t = np.atleast_1d(values["t"]).astype(float)
    energy = np.atleast_1d(values["E"]).astype(float)
    finite = np.isfinite(energy) & (energy > 0.0)
    diverged = not finite.all()
    stop_time = t[-1]
    return t[finite], energy[finite], diverged, stop_time


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), constrained_layout=True)

    for ax, (title, curves) in zip(axes, PANELS):
        loaded = []
        for name, label, color, style in curves:
            t, energy, diverged, stop_time = load(SCAN / name)
            loaded.append((t, energy, diverged, stop_time))
            ax.semilogy(t, energy, color=color, linestyle=style, linewidth=1.5,
                        marker="o", markersize=3.0, label=label)

        finite_energy = np.concatenate([entry[1] for entry in loaded])
        ylo = 10.0 ** np.floor(np.log10(finite_energy.min()))
        yhi = 10.0 ** np.ceil(np.log10(finite_energy.max()))

        for (t, energy, diverged, stop_time), (_, _, color, _) in zip(loaded, curves):
            if not diverged:
                continue
            ax.axvline(stop_time, color=color, linestyle=":", linewidth=1.3,
                       alpha=0.85)
            ax.annotate(
                "", xy=(stop_time, yhi), xytext=(stop_time, energy[-1]),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3,
                                alpha=0.9),
            )

        # Label the stop times on a ladder so nearby times do not overlap.
        stops = [
            (stop_time, color)
            for (_, _, diverged, stop_time), (_, _, color, _) in zip(loaded, curves)
            if diverged
        ]
        for rank, (stop_time, color) in enumerate(sorted(stops)):
            ax.annotate(
                f"blows up at t = {stop_time:.3f}",
                xy=(stop_time, yhi),
                xytext=(6, -8 - 13 * rank),
                textcoords="offset points",
                fontsize=8, color=color, va="top", ha="left",
            )

        ax.set_ylim(ylo, yhi)
        ax.set_title(title)
        ax.set_xlabel("time $t$")
        ax.set_ylabel("energy $E$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Energy for the Taylor-Green and random runs, log energy axis")
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
