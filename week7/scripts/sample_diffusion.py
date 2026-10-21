#!/usr/bin/env python3
"""Run DDPM reverse sampling from a saved time-conditioned noise predictor."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion import beta_schedule, ddpm_sample, sample_target

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

NUM_STEPS = 1000


def load_params(path: Path, num_layers: int) -> tuple:
    with np.load(path) as saved:
        params = tuple((jnp.asarray(saved[f"layer_{i}_weights"]),
                        jnp.asarray(saved[f"layer_{i}_biases"]))
                       for i in range(num_layers))
        update = int(saved["update"])
    return params, update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/sampling"))
    parser.add_argument("--figure", type=Path, default=Path("evidence/ddpm-samples.png"))
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    config = tomllib.loads(args.config.read_text())
    model_cfg = config["model"]
    checkpoint = args.checkpoint or Path("artifacts/training") / (
        f"checkpoint-update-{int(config['training']['updates']):05d}.npz"
    )
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    params, checkpoint_update = load_params(checkpoint, len(model_cfg["hidden_widths"]) + 1)

    key = jax.random.PRNGKey(args.seed)
    sample_key, target_key = jax.random.split(key)
    betas = beta_schedule(NUM_STEPS, 0.0001, 0.02)
    generated = ddpm_sample(
        sample_key,
        params,
        args.num_samples,
        betas,
        int(model_cfg["time_features"]),
        float(model_cfg["time_frequency_base"]),
        float(model_cfg["time_offset"]),
    )
    target = sample_target(target_key, args.num_samples)

    args.out.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "samples.npz",
        generated=np.asarray(generated),
        target_reference=np.asarray(target),
    )
    settings = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_steps": NUM_STEPS,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "target_centers": [[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
        "target_covariance": [[0.09, 0.0], [0.0, 0.09]],
        "mixture_weights": [0.25, 0.25, 0.25, 0.25],
        "checkpoint": str(checkpoint),
        "checkpoint_update": checkpoint_update,
        "reverse_variance": "beta_t",
        "clip_predicted_x0": False,
        "add_noise_at_final_step": False,
        "jax_backend": jax.default_backend(),
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    shutil.copy2(args.config, args.out / "diffusion.toml")

    target_np = np.asarray(target)
    generated_np = np.asarray(generated)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, points, title in zip(
        axes, (target_np, generated_np), ("Target mixture", f"DDPM (update {checkpoint_update})")
    ):
        ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.4, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    fig.suptitle("Target samples and DDPM reverse samples")
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(f"Saved DDPM samples and settings to {args.out}")
    print(f"Saved figure to {args.figure}")


if __name__ == "__main__":
    main()
