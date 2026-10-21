#!/usr/bin/env python3
"""Plot the recorded per-update noise-prediction MSE history."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("artifacts/training/loss-history.csv"))
    parser.add_argument("--output", type=Path, default=Path("evidence/noise-prediction-loss.png"))
    parser.add_argument("--window", type=int, default=250)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")

    history = np.genfromtxt(args.history, delimiter=",", names=True)
    updates = np.atleast_1d(history["update"])
    losses = np.atleast_1d(history["noise_mse"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    window = min(args.window, len(losses))
    moving_average = np.convolve(losses, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(updates, losses, color="tab:blue", alpha=0.18, linewidth=0.6,
            label="Per-update minibatch MSE")
    ax.plot(updates[window - 1:], moving_average, color="tab:orange", linewidth=2,
            label=f"{window}-update moving average")
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Noise-prediction MSE")
    ax.set_title("Time-conditioned noise predictor training")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved training-loss figure to {args.output}")


if __name__ == "__main__":
    main()
