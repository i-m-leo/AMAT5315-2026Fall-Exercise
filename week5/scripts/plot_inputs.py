#!/usr/bin/env python3
"""Plot the geometry and the source pulse an experiment file defines.

Panel 1: background speed, perturbation, shots, and receivers, with
horizontal position and depth in kilometres.
Panel 2: the Ricker pulse at every step's start, against time in seconds.

Writes artifacts/inputs.png (or the paths given on the command line).
Run from week5/.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    experiment_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "inputs" / "reflector.json"
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "artifacts" / "inputs.png"

    exp = json.loads(experiment_path.read_text())
    nx, nz = exp["nx"], exp["nz"]
    background = np.asarray(exp["background"], dtype=float)
    perturbation = np.asarray(exp["perturbation"], dtype=float)
    shots = np.asarray(exp["shots"], dtype=float)
    receivers = np.asarray(exp["receivers"], dtype=float)

    # Reduced units to physical ones: cells to kilometres, steps to seconds.
    cell_km = exp["dx"] * exp["length_unit_m"] / 1000.0
    speed_scale = (exp["length_unit_m"] / 1000.0) / exp["time_unit_s"]

    # Pixel edges are half a cell outside the first and last cell centres.
    x0, x1 = -cell_km / 2.0, (nx - 0.5) * cell_km
    z0, z1 = -cell_km / 2.0, (nz - 0.5) * cell_km
    extent = (x0, x1, z1, z0)  # depth increases downwards

    fig, (ax_geo, ax_src) = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)

    # ---------------------------------------------------------------- geometry
    limit = float(np.abs(perturbation).max()) * speed_scale
    image = ax_geo.imshow(
        perturbation * speed_scale,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    uniform = background.max() - background.min() < 1e-9 * (1.0 + abs(background.max()))
    if uniform:
        geometry = f"uniform background c = {background[0, 0] * speed_scale:.2f} km/s"
    else:
        geometry = "background contours (c) over the perturbation"
        ax_geo.contour(
            np.arange(nx) * cell_km,
            np.arange(nz) * cell_km,
            background * speed_scale,
            levels=8,
            colors="k",
            linewidths=0.4,
            alpha=0.5,
        )
    fig.colorbar(image, ax=ax_geo, shrink=0.85, pad=0.02, label="velocity perturbation (km/s)")

    ax_geo.plot(
        shots[:, 0] * cell_km,
        shots[:, 1] * cell_km,
        "*",
        ms=16,
        mfc="lime",
        mec="black",
        mew=0.8,
        ls="none",
        label=f"shots ({len(shots)})",
    )
    ax_geo.plot(
        receivers[:, 0] * cell_km,
        receivers[:, 1] * cell_km,
        "v",
        ms=6,
        mfc="black",
        mec="white",
        mew=0.7,
        ls="none",
        label=f"receivers ({len(receivers)})",
    )
    ax_geo.legend(loc="upper right", framealpha=0.9)
    ax_geo.set_title(f"Geometry: {geometry}")
    ax_geo.set_xlabel("x (km)")
    ax_geo.set_ylabel("depth (km)")

    # ------------------------------------------------------------ source pulse
    time_unit = exp["time_unit_s"]
    f0 = exp["source_frequency"] / time_unit  # Hz
    t0 = exp["source_peak_time"] * time_unit  # s
    dt = exp["dt"] * time_unit  # s
    times = np.arange(exp["steps"]) * dt
    a = (np.pi * f0 * (times - t0)) ** 2
    pulse = exp["source_amplitude"] * (1.0 - 2.0 * a) * np.exp(-a)

    ax_src.plot(times, pulse, "-", lw=1.0, color="crimson")
    ax_src.plot(times, pulse, "o", ms=2.2, color="crimson")
    ax_src.axvline(t0, color="gray", ls=":", lw=1.0)
    peak = times[int(np.argmax(pulse))]
    ax_src.annotate(
        f"peak {pulse.max():.3g} at t = {peak:.2f} s (t0 = {t0:.2f} s)",
        (peak, pulse.max()),
        xytext=(10, -14),
        textcoords="offset points",
    )
    ax_src.set_title(f"Source pulse: Ricker f0 = {f0:.2f} Hz, t0 = {t0:.2f} s")
    ax_src.set_xlabel("time (s)")
    ax_src.set_ylabel("amplitude")
    ax_src.set_xlim(0.0, times[-1])
    ax_src.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
