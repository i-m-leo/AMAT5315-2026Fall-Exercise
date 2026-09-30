#!/usr/bin/env python3
"""Plot the forward run's receiver gathers, wavefield, and echo.

One gather per shot: receiver position in kilometres across, time in seconds
down, pressure as colour. Also prints the L2 norm of all traces and each
shot's largest absolute pressure with its trace index and receiver.

Reads artifacts/forward, writes gathers.png and wavefield.png there.
Run from week5/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "forward"


def cell_edges(centres):
    """Cell edges of uniformly or sparsely spaced centres, for pcolormesh."""
    centres = np.asarray(centres, dtype=float)
    if centres.size == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    mid = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate(
        ([centres[0] - (mid[0] - centres[0])], mid, [centres[-1] + (centres[-1] - mid[-1])])
    )


def plot_gathers(exp, traces):
    """One shot gather per shot, positions in km and time in seconds."""
    receivers = np.asarray(exp["receivers"], dtype=float)
    x = receivers[:, 0] * exp["dx"] * exp["length_unit_m"] / 1000.0
    # Row n holds u^{n+1}, the pressure after step n+1, at time (n+1) dt.
    times = (np.arange(traces.shape[1]) + 1) * exp["dt"] * exp["time_unit_s"]
    x_edges, t_edges = cell_edges(x), cell_edges(times)

    n_shots = traces.shape[0]
    fig, axes = plt.subplots(
        1, n_shots, figsize=(4.4 * n_shots, 5.2), sharey=True, constrained_layout=True
    )
    scale = float(np.abs(traces).max())
    for shot, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            x_edges, t_edges, traces[shot], cmap="seismic", vmin=-scale, vmax=scale
        )
        ax.set_ylim(t_edges[-1], t_edges[0])  # time increases downwards
        ax.set_xlabel("receiver position (km)")
        ax.set_title(f"shot {shot}")
        fig.colorbar(mesh, ax=ax, shrink=0.85, pad=0.02, label="pressure")
    axes[0].set_ylabel("time (s)")
    fig.suptitle("Forward receiver gathers (background model)")
    fig.savefig(OUT / "gathers.png", dpi=150)
    plt.close(fig)


def report(exp, traces):
    """L2 norm of all traces, and each shot's largest absolute pressure."""
    norm = float(np.sqrt((traces**2).sum()))
    print(f"L2 norm of all traces: {norm:.12e}")
    for shot in range(traces.shape[0]):
        flat = np.abs(traces[shot]).reshape(-1)
        index = int(flat.argmax())
        trace, receiver = divmod(index, traces.shape[2])
        position = exp["receivers"][receiver][0] * exp["dx"] * exp["length_unit_m"] / 1000.0
        print(
            f"shot {shot}: max |p| = {traces[shot, trace, receiver]:.12e} "
            f"at trace index {trace}, receiver {receiver} (x = {position:.2f} km)"
        )


def plot_wavefield(run, wavefield, echo):
    """The first shot's total wavefield and its reflector echo at three times."""
    times = np.asarray(run["recording"]["times"])
    frames = [len(times) // 4, len(times) // 2, -1]
    fig, axes = plt.subplots(2, len(frames), figsize=(10, 6), constrained_layout=True)
    for col, frame in enumerate(frames):
        for row, (field, label) in enumerate([(wavefield, "wavefield"), (echo, "echo")]):
            ax = axes[row, col]
            scale = np.abs(field[frame]).max()
            im = ax.imshow(field[frame], cmap="seismic", vmin=-scale, vmax=scale)
            ax.set_title(f"{label}, t = {times[frame]:.2f} s")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("First shot: total wavefield and reflector echo")
    fig.savefig(OUT / "wavefield.png", dpi=150)
    plt.close(fig)


def main():
    run = json.loads((OUT / "run.json").read_text())
    traces = np.load(OUT / "traces.npy")
    wavefield = np.load(OUT / "wavefield.npy")
    echo = np.load(OUT / "echo.npy")

    report(run["experiment"], traces)
    plot_gathers(run["experiment"], traces)
    plot_wavefield(run, wavefield, echo)
    print(f"wrote {OUT}/gathers.png and {OUT}/wavefield.png")


if __name__ == "__main__":
    main()
