#!/usr/bin/env python3
"""Plot binarized MNIST as Ising spins and visualize a trained RBM."""
from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def load_binarized_mnist(path: Path, count: int) -> np.ndarray:
    with gzip.open(path / "train-images-idx3-ubyte.gz", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("invalid MNIST image IDX magic")
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    if count < 1 or count > n or pixels.size != n * rows * cols:
        raise ValueError("invalid requested MNIST image count or image file")
    # RBM binary units v in {0,1} map to Ising spins s=2v-1 in {-1,+1}.
    return (2 * (pixels[:count * rows * cols].reshape(count, rows, cols) >= 128).astype(np.int8) - 1)


def spin_figure(spins: np.ndarray, output: Path) -> None:
    columns = min(8, len(spins))
    rows = (len(spins) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.7, rows * 1.9),
                             squeeze=False, layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= len(spins):
            continue
        ax.imshow(spins[i], cmap="gray", vmin=-1, vmax=1,
                  interpolation="nearest", origin="upper")
        ax.set_title(f"sample {i + 1}", fontsize=9)
    fig.suptitle("Binarized MNIST as Ising spins (−1 = black, +1 = white; 28×28)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def rbm_figure(parameters: Path, output: Path) -> None:
    with np.load(parameters) as data:
        w = data["weights"]
        vb = data["visible_bias"]
        hb = data["hidden_bias"]
    if w.shape != (784, 128) or vb.shape != (784,) or hb.shape != (128,):
        raise ValueError(f"unexpected RBM parameter shapes: {w.shape}, {vb.shape}, {hb.shape}")

    fig = plt.figure(figsize=(13, 7), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=(1.15, 2.8), height_ratios=(1, 1))
    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    v_y = np.linspace(0.08, 0.92, 14)
    h_y = np.linspace(0.08, 0.92, 10)
    v_x, h_x = 0.18, 0.80
    # Illustrative links indicate dense bipartite connectivity; weights are shown at right.
    for yv in v_y:
        for yh in h_y:
            ax.plot([v_x + 0.04, h_x - 0.04], [yv, yh], color="#6b7280", alpha=0.12, lw=0.5, zorder=1)
    for y in v_y:
        ax.add_patch(Circle((v_x, y), 0.025, facecolor="#377eb8", edgecolor="white", lw=0.4, zorder=2))
    for y in h_y:
        ax.add_patch(Circle((h_x, y), 0.035, facecolor="#e41a1c", edgecolor="white", lw=0.4, zorder=2))
    ax.text(v_x, 0.99, "Visible layer\n784 binary units", ha="center", va="bottom", color="#245a80")
    ax.text(h_x, 0.99, "Hidden layer\n128 binary units", ha="center", va="bottom", color="#a41414")
    ax.text(0.49, 0.02, "Fully connected bipartite\n(no within-layer couplings)", ha="center", va="bottom", fontsize=10)
    ax.text(v_x, 0.01, f"visible bias bᵢ\nrange [{vb.min():.2f}, {vb.max():.2f}]", ha="center", va="top", fontsize=8)
    ax.text(h_x, 0.01, f"hidden bias cⱼ\nrange [{hb.min():.2f}, {hb.max():.2f}]", ha="center", va="top", fontsize=8)

    axw = fig.add_subplot(gs[0, 1])
    im = axw.imshow(w, cmap="coolwarm", aspect="auto", interpolation="nearest", vmin=-np.max(np.abs(w)), vmax=np.max(np.abs(w)))
    axw.set_title("Learned couplings Wᵢⱼ (all 784 × 128 weights)")
    axw.set_xlabel("Hidden unit j"); axw.set_ylabel("Visible pixel i (row-major)")
    fig.colorbar(im, ax=axw, label="Weight")

    axh = fig.add_subplot(gs[1, 1])
    axh.hist(w.ravel(), bins=60, color="#756bb1", alpha=0.85)
    axh.axvline(0, color="black", lw=0.8)
    axh.set_title(f"Coupling distribution (mean={w.mean():.3g}, std={w.std():.3g})")
    axh.set_xlabel("Weight Wᵢⱼ"); axh.set_ylabel("Number of couplings")
    fig.suptitle("MNIST restricted Boltzmann machine — CD-1 learned parameters")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("inputs/mnist"))
    parser.add_argument("--parameters", type=Path, default=Path("artifacts/rbm/cd1-parameters.npz"))
    parser.add_argument("--out", type=Path, default=Path("evidence"))
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    spin_figure(load_binarized_mnist(args.data, args.count), args.out / "mnist-spin-configurations.png")
    rbm_figure(args.parameters, args.out / "rbm-visible-hidden-couplings.png")


if __name__ == "__main__":
    main()
