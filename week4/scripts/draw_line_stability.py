"""Draw the week4 stability figure.

Top row: the complex plane of z = lambda h, coloured by the per-step growth
factor measured by the library integrators (`dump_stability`), on a log scale,
with the |R(z)| = 1 curves of all three schemes and the spectral modes of the
line at nu = 0.05, n = 64, c = 1 for dt = 0.045 and dt = 0.056.

Bottom row: u(x, t) for a periodic Gaussian pulse (sigma = 0.35) centred at
x = pi/2, integrated by the library RK4 to t = 6 at the same dt values. x runs
across, t runs downward.

Run from week4/ after the evidence files exist.
"""

import gzip
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
GRID = EVIDENCE / "stability-grid.txt.gz"
OUT = EVIDENCE / "line-stability.png"

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
N = 64
C = 1.0
NU = 0.05
DTS = [0.045, 0.056]


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


def load_pulse(dt):
    path = EVIDENCE / f"pulse-dt{dt:g}.txt"
    header, values = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                header = line
                continue
            values.append([float(v) for v in line.split()])
    values = np.asarray(values)
    times = values[:, 0]
    field = values[:, 1:]
    return times, field, (header or "").strip()


def draw_stability(axes, fig):
    data = load_grid()
    ks = np.fft.fftfreq(N, d=1.0 / N)
    lambda_spectral = -NU * ks**2 - 1j * C * ks

    for ax, name in zip(axes, SCHEMES):
        re, im, g = to_square(data[name])
        mesh = ax.pcolormesh(
            re, im, g.T, shading="auto", cmap="viridis",
            norm=LogNorm(vmin=1e-6, vmax=1e2), rasterized=True,
        )
        fig.colorbar(mesh, ax=ax, label="growth factor per step  |y1|")

        xr = np.linspace(-3.5, 3.5, 700)
        xg, yg = np.meshgrid(xr, xr)
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

        for dt, marker in zip(DTS, ("o", "^")):
            z = lambda_spectral * dt
            ax.scatter(z.real, z.imag, s=8, marker=marker, color="black",
                       linewidths=0.0, zorder=4)
            ax.scatter(z.real, z.imag, s=30, marker=marker, facecolors="none",
                       edgecolors="white", linewidths=1.0, zorder=5)

        ax.set_title(LABELS[name])
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.axhline(0.0, color="0.8", linewidth=0.6, zorder=0)
        ax.axvline(0.0, color="0.8", linewidth=0.6, zorder=0)

        handles = [
            plt.Line2D([], [], color=COLORS[s], linestyle="-",
                       label=f"|R|=1 {LABELS[s]}")
            for s in SCHEMES
        ]
        handles += [
            plt.Line2D([], [], marker="o", linestyle="none",
                       markerfacecolor="none", markeredgecolor="white",
                       label="dt = 0.045"),
            plt.Line2D([], [], marker="^", linestyle="none",
                       markerfacecolor="none", markeredgecolor="white",
                       label="dt = 0.056"),
        ]
        if name == "classical-rk4":
            ax.legend(handles=handles, loc="lower right", fontsize=8,
                      framealpha=0.9)


def draw_pulse(axes, fig):
    for ax, dt in zip(axes, DTS):
        times, field, header = load_pulse(dt)
        peak = np.max(np.abs(field))
        im = ax.imshow(
            field, origin="upper", aspect="auto",
            extent=[0.0, 2.0 * np.pi, times[-1], 0.0],
            cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        )
        fig.colorbar(im, ax=ax, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel("t  (downward)")
        ax.set_title(f"RK4 pulse, dt = {dt:g}   (peak |u| = {peak:.2e})")
        if peak > 1.5:
            ax.text(0.5, 0.06, "unstable", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11, color="black",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))


def main():
    fig = plt.figure(figsize=(17.0, 10.6), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0])

    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    draw_stability(top, fig)

    bottom = [fig.add_subplot(gs[1, j]) for j in range(2)]
    draw_pulse(bottom, fig)
    hidden = fig.add_subplot(gs[1, 2])
    hidden.axis("off")

    fig.suptitle(
        "Week 4 stability: growth factor on the complex plane (top) and "
        f"RK4 pulse u(x, t) (bottom), nu = {NU}, n = {N}, c = {C}",
        fontsize=13,
    )
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
