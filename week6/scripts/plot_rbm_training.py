#!/usr/bin/env python3
"""Plot CD surrogate and held-out one-step changed-pixel rate from metrics.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training(metrics_path: Path, output: Path) -> None:
    metrics = json.loads(metrics_path.read_text())
    models = metrics["models"]
    if not models:
        raise ValueError("metrics file contains no models")
    fig, (ax_cd, ax_rec) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = {1: "#1f77b4", 10: "#d95f02"}
    for model in models:
        k = int(model["cd_steps"])
        history = model["history"]
        epochs = np.asarray([point["epoch"] for point in history])
        if len(epochs) < 2 or epochs[0] != 0:
            raise ValueError(f"CD-{k} history must contain initialization and epoch records")
        # CD surrogate is the mean of batch surrogates before that epoch's updates;
        # therefore it is associated with the post-epoch model, at epochs 1..N.
        cd_epochs = epochs[1:]
        surrogate = np.asarray([point["cd_surrogate"] for point in history[1:]])
        rec_epochs = epochs
        reconstruction = np.asarray([point["reconstruction_error_cd1"] for point in history])
        color = colors.get(k)
        ax_cd.plot(cd_epochs, surrogate, marker="o", ms=3, lw=1.7,
                   label=f"CD-{k}", color=color)
        ax_rec.plot(rec_epochs, reconstruction, marker="o", ms=3, lw=1.7,
                    label=f"CD-{k}", color=color)
    ax_cd.set(title="Training CD surrogate", xlabel="Epoch", ylabel="Mean batch surrogate")
    ax_cd.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax_rec.set(title="Held-out one-step reconstruction", xlabel="Epoch",
               ylabel="Fraction of changed pixels")
    ax_rec.set_ylim(bottom=0)
    for ax in (ax_cd, ax_rec):
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("MNIST RBM training: CD-1 vs CD-10")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/rbm/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("evidence/rbm-training.png"))
    args = parser.parse_args()
    plot_training(args.metrics, args.output)


if __name__ == "__main__":
    main()
