#!/usr/bin/env python3
"""Plot the forward run's gather, wavefield, and echo.

Reads artifacts/forward/*.npy, writes gathers.png, wavefield.png, and
echo.png there. Run from week5/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "forward"


def main():
    import json

    run = json.loads((OUT / "run.json").read_text())
    traces = np.load(OUT / "traces.npy")
    wavefield = np.load(OUT / "wavefield.npy")
    echo = np.load(OUT / "echo.npy")
    dt = run["experiment"]["dt"]
    times = np.asarray(run["recording"]["times"])

    # Gathers: one panel per shot, receiver axis across, time down.
    fig, axes = plt.subplots(1, traces.shape[0], figsize=(9, 5), sharey=True)
    t = np.arange(traces.shape[1]) * dt
    for shot, ax in enumerate(axes):
        im = ax.imshow(
            traces[shot],
            aspect="auto",
            origin="upper",
            extent=[0, traces.shape[2] - 1, t[-1], t[0]],
            cmap="seismic",
        )
        ax.set_xlabel("receiver index")
        ax.set_title(f"shot {shot}")
        fig.colorbar(im, ax=ax, shrink=0.8, label="pressure")
    axes[0].set_ylabel("time (units)")
    fig.suptitle("Forward gathers (background model)")
    fig.tight_layout()
    fig.savefig(OUT / "gathers.png", dpi=150)
    plt.close(fig)

    # Wavefield and echo at three frames.
    frames = [len(times) // 4, len(times) // 2, -1]
    fig, axes = plt.subplots(2, len(frames), figsize=(10, 6))
    for col, frame in enumerate(frames):
        for row, (field, label) in enumerate([(wavefield, "wavefield"), (echo, "echo")]):
            ax = axes[row, col]
            scale = np.abs(field[frame]).max()
            im = ax.imshow(field[frame], cmap="seismic", vmin=-scale, vmax=scale)
            ax.set_title(f"{label}, t = {times[frame]:.1f}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("First shot: total wavefield and reflector echo")
    fig.tight_layout()
    fig.savefig(OUT / "wavefield.png", dpi=150)
    plt.close(fig)

    print(f"wrote {OUT}/gathers.png and {OUT}/wavefield.png")


if __name__ == "__main__":
    main()
