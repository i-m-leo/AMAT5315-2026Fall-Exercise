"""Draw the complex plane of z = lambda h for the week4 integrators.

Reads the measured per-step growth factor dumped by `dump_stability` and draws
it on a log colour scale, overlays the |R(z)| = 1 stability curves of the three
integrators, and marks the spectral modes of the advection-diffusion line at
nu = 0.05, n = 64, c = 1 for the steps dt = 0.045 and dt = 0.056.

Run from week4/ after `cargo run --bin dump-stability -- 3.5 321 | gzip -9 >
evidence/stability-grid.txt.gz` wrote the grid.
"""

import gzip
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "evidence" / "stability-grid.txt.gz"
OUT = ROOT / "evidence" / "linestability.png"

SCHEMES = ["forward-euler", "explicit-midpoint", "classical-rk4"]
LABELS = {
    "forward-euler": "forward Euler",
    "explicit-midpoint": "explicit midpoint",
    "classical-rk4": "classical RK4",
}
COLORS = {
    "forward-euler": "#d62728",
    "explicit-midpoint": "#2ca02c",
    "classical-rk4": "#1f77b4",
}


def stability_polynomial(name, z):
    if name == "forward-euler":
        return 1.0 + z
    if name == "explicit-midpoint":
        return 1.0 + z + z**2 / 2.0
    if name == "classical-rk4":
        return 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
    raise ValueError(name)


def load_grid():
    data = {}
    with gzip.open(GRID, "rt") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            name, re, im, g = line.split()
            data.setdefault(name, []).append((float(re), float(im), float(g)))
    return data


def to_square(rows):
    rows = np.asarray(rows)
    re = np.unique(rows[:, 0])
    im = np.unique(rows[:, 1])
    g = rows[:, 2].reshape(re.size, im.size)
    return re, im, g


def main():
    data = load_grid()
    n = 64
    c = 1.0
    nu = 0.05
    ks = np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers
    lambda_spectral = -nu * ks**2 - 1j * c * ks

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)
    for ax, name in zip(axes, SCHEMES):
        re, im, g = to_square(data[name])
        mesh = ax.pcolormesh(
            re, im, g.T, shading="auto", cmap="viridis",
            norm=LogNorm(vmin=1e-6, vmax=1e2), rasterized=True,
        )
        fig.colorbar(mesh, ax=ax, label="growth factor per step  |y1|")

        # |R(z)| = 1 stability curve for every integrator, on this panel.
        xr = np.linspace(-3.5, 3.5, 700)
        yr = np.linspace(-3.5, 3.5, 700)
        xg, yg = np.meshgrid(xr, yr)
        zg = xg + 1j * yg
        for other in SCHEMES:
            mod = np.abs(stability_polynomial(other, zg))
            ax.contour(
                xg, yg, mod, levels=[1.0],
                colors=[COLORS[other]],
                linewidths=1.6 if other == name else 1.0,
                linestyles="-" if other == name else "--",
                alpha=1.0 if other == name else 0.7,
            )

        # Spectral modes of the line, one dot per step.
        for dt, marker in ((0.045, "o"), (0.056, "^")):
            z = lambda_spectral * dt
            ax.scatter(
                z.real, z.imag, s=8, marker=marker,
                color="black", linewidths=0.0, zorder=4,
            )
            ax.scatter(
                z.real, z.imag, s=30, marker=marker,
                facecolors="none", edgecolors="white", linewidths=1.0, zorder=5,
            )

        ax.set_title(LABELS[name])
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.axhline(0.0, color="0.8", linewidth=0.6, zorder=0)
        ax.axvline(0.0, color="0.8", linewidth=0.6, zorder=0)

    handles = [
        plt.Line2D([], [], color=COLORS[s], linestyle="-", label=f"|R|=1 {LABELS[s]}")
        for s in SCHEMES
    ]
    handles += [
        plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="white", label="dt = 0.045"),
        plt.Line2D([], [], marker="^", linestyle="none", markerfacecolor="none",
                   markeredgecolor="white", label="dt = 0.056"),
    ]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Step growth factor on the complex plane, with |R(z)| = 1 and "
        f"line modes (nu={nu}, n={n}, c={c})",
        fontsize=12,
    )
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
