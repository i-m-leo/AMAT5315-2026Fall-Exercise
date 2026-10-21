#!/usr/bin/env python3
"""Compare selected-step deterministic DDIM with a mixture-oracle predictor."""
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

from diffusion import (
    CENTERS,
    TARGET_VARIANCE,
    beta_schedule,
    ddim_trajectory,
    mixture_conditional_mean_noise,
    predict_noise,
    sample_target,
    select_ddim_timesteps,
)
from mode_metrics import assign_nearest_mode, summarize_modes

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

TOTAL_STEPS = 1000
DEFAULT_SAMPLES = 2000


def load_checkpoint(path: Path, expected_update: int, num_layers: int) -> tuple:
    with np.load(path) as saved:
        update = int(saved["update"])
        params = tuple((jnp.asarray(saved[f"layer_{i}_weights"]),
                        jnp.asarray(saved[f"layer_{i}_biases"]))
                       for i in range(num_layers))
    if update != expected_update:
        raise ValueError(f"{path} stores update {update}, expected {expected_update}")
    return params


def explicit_timesteps(value: str) -> np.ndarray:
    try:
        levels = np.asarray([int(part.strip()) for part in value.split(",")], dtype=np.int32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("levels must be comma-separated integers") from exc
    if len(levels) == 0 or levels[0] != TOTAL_STEPS:
        raise argparse.ArgumentTypeError("explicit levels must start at 1000")
    if np.any(levels < 1) or np.any(levels > TOTAL_STEPS) or np.any(np.diff(levels) >= 0):
        raise argparse.ArgumentTypeError("levels must be strictly descending values in [1, 1000]")
    return levels


def _plot_distribution(ax, samples: np.ndarray, centers: np.ndarray,
                       title: str, coordinate_limit: float) -> None:
    labels = assign_nearest_mode(samples, centers)
    palette = plt.get_cmap("tab10").colors
    for mode in range(len(centers)):
        points = samples[labels == mode]
        ax.scatter(points[:, 0], points[:, 1], s=7, alpha=0.38,
                   color=palette[mode], linewidths=0,
                   label=f"Mode {mode + 1} ({len(points)})")
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", c="black", s=70,
               linewidths=1.4, label="Reference centres", zorder=4)
    ax.set_title(title)
    ax.set_xlim(-coordinate_limit, coordinate_limit)
    ax.set_ylim(-coordinate_limit, coordinate_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=7, ncols=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/training/checkpoint-update-25000.npz"))
    parser.add_argument("--num-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=5316)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/ddim-comparison"))
    parser.add_argument("--figure", type=Path, default=Path("evidence/ddim-comparison.png"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--inference-steps", type=int, default=50,
                           help="number of evenly spaced DDIM levels (default: 50)")
    selection.add_argument("--levels", type=explicit_timesteps,
                           help="explicit descending levels, e.g. 1000,900,800,...,1")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    config = tomllib.loads(args.config.read_text())
    model_config = config["model"]
    with np.load(args.checkpoint) as checkpoint:
        checkpoint_update = int(checkpoint["update"])
    params = load_checkpoint(args.checkpoint, checkpoint_update,
                             len(model_config["hidden_widths"]) + 1)
    if args.levels is None and not 1 <= args.inference_steps <= TOTAL_STEPS:
        parser.error(f"--inference-steps must be in [1, {TOTAL_STEPS}]")
    levels = (jnp.asarray(args.levels, dtype=jnp.int32) if args.levels is not None
              else select_ddim_timesteps(args.inference_steps, TOTAL_STEPS))
    selected_levels = np.asarray(levels, dtype=np.int32)

    betas = beta_schedule(TOTAL_STEPS, 0.0001, 0.02)
    root_key = jax.random.PRNGKey(args.seed)
    initial_key, target_key = jax.random.split(root_key)
    initial = jax.random.normal(initial_key, (args.num_samples, 2), dtype=jnp.float32)
    target = np.asarray(sample_target(target_key, args.num_samples))

    learned_predictor = lambda x_t, level: predict_noise(
        params,
        x_t,
        level,
        int(model_config["time_features"]),
        float(model_config["time_frequency_base"]),
        float(model_config["time_offset"]),
    )
    oracle_predictor = lambda x_t, level: mixture_conditional_mean_noise(
        x_t, level, betas, CENTERS, TARGET_VARIANCE
    )
    learned_path = np.asarray(jax.jit(lambda x: ddim_trajectory(
        x, levels, betas, learned_predictor
    ))(initial))
    oracle_path = np.asarray(jax.jit(lambda x: ddim_trajectory(
        x, levels, betas, oracle_predictor
    ))(initial))
    trained_samples = learned_path[-1]
    oracle_samples = oracle_path[-1]
    centers = np.asarray(CENTERS)
    stats = {}
    for name, samples in (("target_reference", target),
                          ("trained_predictor", trained_samples),
                          ("exact_conditional_mean_noise", oracle_samples)):
        labels = assign_nearest_mode(samples, centers)
        stats[name] = summarize_modes(samples, centers, labels)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "ddim-comparison.npz",
        levels=selected_levels,
        betas=np.asarray(betas),
        initial_gaussian_samples=np.asarray(initial),
        target_samples=target,
        trained_trajectory=learned_path,
        oracle_trajectory=oracle_path,
    )
    settings = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "total_training_noise_levels": TOTAL_STEPS,
        "selected_descending_levels": selected_levels.tolist(),
        "num_ddim_steps": len(selected_levels),
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": checkpoint_update,
        "sampler": "deterministic DDIM (eta=0)",
        "same_initial_gaussian_samples_for_both_predictors": True,
        "reverse_random_noise": "none; DDIM is deterministic",
        "prediction_methods": ["trained noise predictor", "exact mixture E[epsilon|x_t]"],
        "target_centers": centers.tolist(),
        "target_covariance": [[TARGET_VARIANCE, 0.0], [0.0, TARGET_VARIANCE]],
        "target_mixture_weights": [0.25] * 4,
        "mode_assignment": "nearest reference centre",
        "spread_convention": "population variance and standard deviation (ddof=0)",
        "jax_backend": jax.default_backend(),
        "figure": str(args.figure),
    }
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.out_dir / "mode-statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    shutil.copy2(args.config, args.out_dir / "diffusion.toml")

    all_points = np.concatenate((target, trained_samples, oracle_samples), axis=0)
    coordinate_limit = max(5.0, float(np.max(np.abs(all_points))) * 1.05)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    _plot_distribution(axes[0], target, centers, "Target mixture", coordinate_limit)
    _plot_distribution(
        axes[1], trained_samples, centers,
        f"Trained predictor DDIM ({len(selected_levels)} steps)", coordinate_limit,
    )
    _plot_distribution(
        axes[2], oracle_samples, centers,
        f"Exact conditional-mean DDIM ({len(selected_levels)} steps)", coordinate_limit,
    )
    fig.suptitle("Deterministic DDIM: learned predictor vs exact mixture conditional mean")
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)

    print(f"Saved {len(selected_levels)}-step DDIM comparison to {args.out_dir}")
    print(f"Selected levels: {selected_levels.tolist()}")
    print(f"Saved distribution comparison to {args.figure}")
    for method in ("trained_predictor", "exact_conditional_mean_noise"):
        print(f"{method} mode populations and empirical centres:")
        for row in stats[method]:
            print(
                f"  {row['reference_center']}: n={row['population_count']} "
                f"({100 * row['population_fraction']:.1f}%), "
                f"centre={np.round(row['empirical_center'], 3).tolist()}, "
                f"SD={np.round(row['coordinate_std'], 3).tolist()}"
            )


if __name__ == "__main__":
    main()
