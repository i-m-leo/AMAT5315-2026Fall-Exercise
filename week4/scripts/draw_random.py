"""Draw the vorticity of the random run at t = 0, 2, 5, and 10.

One row of four panels on a shared diverging colour scale, symmetric about
zero. Reads artifacts/random/fields.jsonl and writes evidence/random.png.
Run from week4/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "random"
FIELDS = ARTIFACTS / "fields.jsonl"
OUT = ROOT / "evidence" / "random.png"

TIMES = [0.0, 2.0, 5.0, 10.0]


def load_frames():
    frames = {}
    with FIELDS.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            frames[round(frame["t"], 6)] = frame
    return frames


def main():
    with (ARTIFACTS / "run.json").open() as handle:
        run = json.load(handle)
    n = int(run["n"])

    frames = load_frames()
    for t in TIMES:
        assert t in frames, f"no snapshot at t = {t}"

    grid = np.arange(n) * 2.0 * np.pi / n
    xx, yy = np.meshgrid(grid, grid)

    fields = [np.asarray(frames[t]["omega"], dtype=float).reshape(n, n) for t in TIMES]
    limit = max(np.abs(field).max() for field in fields)

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.8), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, t, field in zip(axes, TIMES, fields):
        mesh = ax.pcolormesh(
            xx, yy, field, shading="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, rasterized=True,
        )
        ax.set_title(f"t = {t:g}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")

    fig.colorbar(mesh, ax=axes, label="vorticity $\\omega$", shrink=0.85)
    fig.suptitle(
        f"Vorticity of the random run, n = {n}, seed = {run['seed']}, "
        f"$\\nu$ = {run['nu']}, $|k|$ in [{run['k_band'][0]:g}, {run['k_band'][1]:g}]"
    )
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}  (shared scale |omega| <= {limit:.3f})")


if __name__ == "__main__":
    main()
