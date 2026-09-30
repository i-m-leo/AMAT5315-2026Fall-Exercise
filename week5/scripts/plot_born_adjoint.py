#!/usr/bin/env python3
"""Plot the Born data and the migrated image.

Reads artifacts/born and artifacts/adjoint, writes born.png and image.png.
Prints the Born data norm and the dot-product identity
<Born data, Born data> = <image, perturbation>. Run from week5/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs" / "reflector.json"


def main():
    exp = json.loads(INPUTS.read_text())
    born = np.load(ROOT / "artifacts" / "born" / "born_data.npy")
    image = np.load(ROOT / "artifacts" / "adjoint" / "image.npy")
    perturbation = np.asarray(exp["perturbation"], dtype=float)

    cell_km = exp["dx"] * exp["length_unit_m"] / 1000.0
    x = np.arange(exp["nx"])[:, None] * cell_km
    z = np.arange(exp["nz"])[:, None] * cell_km

    # Born data: one gather per shot, receiver across, time down.
    times = (np.arange(born.shape[1]) + 1) * exp["dt"] * exp["time_unit_s"]
    fig, axes = plt.subplots(1, born.shape[0], figsize=(13, 4.4), sharey=True, constrained_layout=True)
    limit = float(np.abs(born).max())
    # Cell edges, one more than the number of samples along each axis.
    t_edges = np.concatenate(([times[0] - exp["dt"] * exp["time_unit_s"]],
                              0.5 * (times[1:] + times[:-1]),
                              [times[-1] + exp["dt"] * exp["time_unit_s"]]))
    receiver_edges = np.arange(born.shape[2] + 1, dtype=float)
    for shot, ax in enumerate(axes):
        mesh = ax.pcolormesh(receiver_edges, t_edges, born[shot],
                             cmap="seismic", vmin=-limit, vmax=limit)
        ax.set_ylim(t_edges[-1], t_edges[0])
        ax.set_xlabel("receiver index")
        ax.set_title(f"shot {shot}")
        fig.colorbar(mesh, ax=ax, shrink=0.85, label="Born pressure")
    axes[0].set_ylabel("time (s)")
    fig.suptitle("Born scattered data (background wavefield, perturbation velocity)")
    fig.savefig(ROOT / "artifacts" / "born" / "born.png", dpi=150)
    plt.close(fig)

    # The migrated image, and the truth it should focus onto.
    limit = float(np.abs(image).max())
    fig, (ax_image, ax_true) = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    x_edges = np.linspace(-cell_km / 2, (exp["nx"] - 0.5) * cell_km, exp["nx"] + 1)
    z_edges = np.linspace(-cell_km / 2, (exp["nz"] - 0.5) * cell_km, exp["nz"] + 1)
    for ax, field, title in [
        (ax_image, image, "adjoint image (velocity sensitivity)"),
        (ax_true, perturbation, "true perturbation"),
    ]:
        scale = float(np.abs(field).max())
        mesh = ax.pcolormesh(x_edges, z_edges, field, cmap="seismic", vmin=-scale, vmax=scale)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("depth (km)")
        ax.set_ylim(z_edges[-1], z_edges[0])
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.savefig(ROOT / "artifacts" / "adjoint" / "image.png", dpi=150)
    plt.close(fig)

    # The dot-product identity.
    image = np.asarray(image)
    perturbation = perturbation.ravel()
    dot_left = float((born**2).sum())
    dot_right = float((image.ravel() * perturbation).sum())
    profile = np.abs(image).sum(axis=1)
    peak = int(profile.argmax())
    print(f"Born data norm: {np.sqrt(dot_left)!r}")
    print(f"dot-left  <Born, Born>      = {dot_left!r}")
    print(f"dot-right <image, true dv>  = {dot_right!r}")
    print(f"relative error              = {abs(dot_left - dot_right) / abs(dot_left):.3e}")
    print(f"depth profile peaks at z = {peak} ({peak * cell_km:.1f} km)")
    print(f"wrote {ROOT / 'artifacts' / 'born' / 'born.png'}")
    print(f"wrote {ROOT / 'artifacts' / 'adjoint' / 'image.png'}")


if __name__ == "__main__":
    main()
