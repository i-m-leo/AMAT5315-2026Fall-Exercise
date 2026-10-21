#!/usr/bin/env python3
"""Sample the Gaussian-mixture target and illustrate its 1000-step forward process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Use CPU unless the caller explicitly selects a JAX backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion import CENTERS, TARGET_VARIANCE, beta_schedule, forward_diffusion, sample_target

NUM_STEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
CHECKPOINTS = (0, 1, 10, 100, 500, NUM_STEPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forward-diffusion"))
    parser.add_argument("--figure", type=Path, default=Path("evidence/forward-diffusion.png"))
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    key = jax.random.PRNGKey(args.seed)
    sample_key, diffusion_key = jax.random.split(key)
    x0 = sample_target(sample_key, args.num_samples)
    betas = beta_schedule(NUM_STEPS, BETA_START, BETA_END)
    trajectory = forward_diffusion(diffusion_key, x0, betas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        x0=np.asarray(x0),
        trajectory=np.asarray(trajectory),
        betas=np.asarray(betas),
        checkpoints=np.asarray(CHECKPOINTS, dtype=np.int32),
    )
    settings = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_steps": NUM_STEPS,
        "beta_start": BETA_START,
        "beta_end": BETA_END,
        "target_centers": np.asarray(CENTERS).tolist(),
        "target_covariance": [[TARGET_VARIANCE, 0.0], [0.0, TARGET_VARIANCE]],
        "mixture_weights": [0.25] * 4,
        "checkpoints": list(CHECKPOINTS),
        "jax_backend": jax.default_backend(),
        "forward_process_only": True,
    }
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, level in zip(axes.flat, CHECKPOINTS):
        points = np.asarray(trajectory[level])
        ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.35, linewidths=0)
        ax.set_title(f"t = {level}")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    fig.suptitle("Four-Gaussian target under forward diffusion")
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(f"Saved raw trajectory and settings to {args.output_dir}")
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
