#!/usr/bin/env python3
"""Plot the recorded flow-matching velocity-loss history."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("artifacts/flow-training/loss-history.csv"))
    parser.add_argument("--output", type=Path, default=Path("evidence/flow-training-loss.png"))
    parser.add_argument("--window", type=int, default=250)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    history = np.genfromtxt(args.history, delimiter=",", names=True)
    updates = np.atleast_1d(history["update"])
    losses = np.atleast_1d(history["velocity_mse"])
    window = min(args.window, len(losses))
    moving = np.convolve(losses, np.ones(window) / window, mode="valid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(updates, losses, color="tab:green", alpha=0.18, linewidth=0.6,
            label="Per-update minibatch MSE")
    ax.plot(updates[window - 1:], moving, color="tab:orange", linewidth=2,
            label=f"{window}-update moving average")
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel("Velocity-prediction MSE")
    ax.set_title("Flow-matching training loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved flow loss figure to {args.output}")


if __name__ == "__main__":
    main()
