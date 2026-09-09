#!/usr/bin/env python3
"""Generate field.png from the md crate's Lennard-Jones functions.

Run from the repository root with:
    ~/anaconda3/bin/python week2/plot_field.py
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


WEEK_DIR = Path(__file__).resolve().parent
MANIFEST = WEEK_DIR / "md" / "Cargo.toml"
OUTPUT = WEEK_DIR / "field.png"
os.environ.setdefault("MPLCONFIGDIR", str(WEEK_DIR / "md" / "target" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def rust_field_data() -> tuple[np.ndarray, np.ndarray, float]:
    """Run the Rust sampler and parse its energy grid and force vectors."""
    cargo = Path("/opt/homebrew/bin/cargo")
    command = str(cargo) if cargo.exists() else "cargo"
    result = subprocess.run(
        [command, "run", "--quiet", "--manifest-path", str(MANIFEST), "--example", "field_data"],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = iter(result.stdout.splitlines())
    _, size_text, limit_text = next(lines).split(",")
    size, limit = int(size_text), float(limit_text)
    energy = np.empty((size, size))
    arrows = []
    for line in lines:
        fields = line.split(",")
        if fields[0] == "e":
            energy[int(fields[1]), int(fields[2])] = float(fields[3])
        elif fields[0] == "f":
            arrows.append(tuple(map(float, fields[1:])))
    return energy, np.asarray(arrows), limit


def main() -> None:
    energy, arrows, limit = rust_field_data()
    x, y, force_x, force_y = arrows.T
    magnitude = np.hypot(force_x, force_y)
    display_length = 0.10 + 0.22 * np.log1p(magnitude) / np.log1p(magnitude.max())
    arrow_x = force_x / magnitude * display_length
    arrow_y = force_y / magnitude * display_length

    figure, axes = plt.subplots(figsize=(9, 9), dpi=100, layout="constrained")
    image = axes.imshow(
        np.clip(energy, -1.0, 2.0),
        extent=(-limit, limit, -limit, limit),
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=2.0,
        interpolation="bilinear",
    )
    axes.quiver(
        x,
        y,
        arrow_x,
        arrow_y,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="black",
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        zorder=3,
    )
    axes.add_patch(
        Circle(
            (0.0, 0.0),
            0.11,
            facecolor="white",
            edgecolor="black",
            linewidth=2.5,
            zorder=4,
        )
    )

    axes.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        aspect="equal",
        xlabel=r"$x / \sigma$",
        ylabel=r"$y / \sigma$",
        title="Lennard-Jones pair energy (color) and radial force (arrows)",
    )
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.03)
    colorbar.set_label(r"$U / \epsilon$")
    figure.savefig(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
