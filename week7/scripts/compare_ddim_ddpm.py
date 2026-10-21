#!/usr/bin/env python3
"""Compare repeated full-step DDIM and two DDPM noise draws from shared x_T."""
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

from diffusion import CENTERS, beta_schedule, ddpm_trajectory, ddim_trajectory, predict_noise, sample_target
from mode_metrics import assign_nearest_mode

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

NUM_STEPS = 1000
NUM_SAMPLES = 2000


def load_checkpoint(path: Path, expected_update: int, num_layers: int) -> tuple:
    with np.load(path) as saved:
        update = int(saved["update"])
        params = tuple((jnp.asarray(saved[f"layer_{i}_weights"]),
                        jnp.asarray(saved[f"layer_{i}_biases"]))
                       for i in range(num_layers))
    if update != expected_update:
        raise ValueError(f"{path} stores update {update}, expected {expected_update}")
    return params


def plot_sample_trajectories(paths: list[np.ndarray], labels: list[str],
                             output: Path) -> float:
    representative = np.linspace(0, NUM_SAMPLES - 1, 10, dtype=int)
    coordinate_limit = max(6.0, max(float(np.max(np.abs(path))) for path in paths) * 1.03)
    palette = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5), constrained_layout=True)
    for ax, path, title in zip(axes, paths, labels):
        for color_index, sample_index in enumerate(representative):
            color = palette[color_index]
            ax.plot(path[:, sample_index, 0], path[:, sample_index, 1],
                    color=color, alpha=0.65, linewidth=0.8)
            ax.scatter(path[0, sample_index, 0], path[0, sample_index, 1],
                       color=color, marker="o", s=17, zorder=3)
            ax.scatter(path[-1, sample_index, 0], path[-1, sample_index, 1],
                       color=color, marker="*", s=45, edgecolors="black",
                       linewidths=0.3, zorder=4)
        ax.set_title(title)
        ax.set_xlim(-coordinate_limit, coordinate_limit)
        ax.set_ylim(-coordinate_limit, coordinate_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(alpha=0.18)
    fig.suptitle("Reverse trajectories for 10 paired samples (○ xT, ★ x0)")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return coordinate_limit


def plot_final_samples(samples: list[np.ndarray], names: list[str],
                       centers: np.ndarray, output: Path) -> None:
    palette = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4.5),
                             constrained_layout=True)
    for ax, points, title in zip(axes, samples, names):
        labels = assign_nearest_mode(points, centers)
        for mode, center in enumerate(centers):
            mode_points = points[labels == mode]
            ax.scatter(mode_points[:, 0], mode_points[:, 1], s=6, alpha=0.35,
                       color=palette[mode], linewidths=0,
                       label=f"Mode {mode + 1}: {len(mode_points)}")
        ax.scatter(centers[:, 0], centers[:, 1], marker="x", c="black", s=55,
                   linewidths=1.2, zorder=4)
        ax.set_title(title)
        ax.set_xlim(-6.0, 6.0)
        ax.set_ylim(-6.0, 6.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(alpha=0.16)
        ax.legend(loc="upper right", fontsize=6)
    fig.suptitle("2000 final samples per run (crosses: target mode centres)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/training/checkpoint-update-25000.npz"))
    parser.add_argument("--seed", type=int, default=5317)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/ddim-ddpm-comparison"))
    parser.add_argument("--trajectory-figure", type=Path,
                        default=Path("evidence/ddim-ddpm-trajectories.png"))
    parser.add_argument("--samples-figure", type=Path,
                        default=Path("evidence/ddim-ddpm-samples.png"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    config = tomllib.loads(args.config.read_text())
    model_config = config["model"]
    with np.load(args.checkpoint) as checkpoint:
        checkpoint_update = int(checkpoint["update"])
    params = load_checkpoint(args.checkpoint, checkpoint_update,
                             len(model_config["hidden_widths"]) + 1)

    root_key = jax.random.PRNGKey(args.seed)
    initial_key, ddpm_a_key, ddpm_b_key, target_key = jax.random.split(root_key, 4)
    initial = jax.random.normal(initial_key, (NUM_SAMPLES, 2), dtype=jnp.float32)
    ddpm_noise_a = jax.random.normal(ddpm_a_key, (NUM_STEPS, NUM_SAMPLES, 2), dtype=jnp.float32)
    ddpm_noise_b = jax.random.normal(ddpm_b_key, (NUM_STEPS, NUM_SAMPLES, 2), dtype=jnp.float32)
    target = np.asarray(sample_target(target_key, NUM_SAMPLES))
    betas = beta_schedule(NUM_STEPS, 0.0001, 0.02)
    levels = jnp.arange(NUM_STEPS, 0, -1, dtype=jnp.int32)

    learned_predictor = lambda x_t, level: predict_noise(
        params,
        x_t,
        level,
        int(model_config["time_features"]),
        float(model_config["time_frequency_base"]),
        float(model_config["time_offset"]),
    )
    ddim_fn = jax.jit(lambda x: ddim_trajectory(x, levels, betas, learned_predictor))
    ddpm_fn = jax.jit(lambda noise: ddpm_trajectory(params, initial, noise, betas,
                                                    int(model_config["time_features"]),
                                                    float(model_config["time_frequency_base"]),
                                                    float(model_config["time_offset"])))
    ddim_a = np.asarray(ddim_fn(initial))
    ddim_b = np.asarray(ddim_fn(initial))
    ddpm_a = np.asarray(ddpm_fn(ddpm_noise_a))
    ddpm_b = np.asarray(ddpm_fn(ddpm_noise_b))
    paths = [ddim_a, ddim_b, ddpm_a, ddpm_b]
    path_names = ["DDIM run A", "DDIM run B", "DDPM noise draw A", "DDPM noise draw B"]

    ddim_difference = np.abs(ddim_a - ddim_b)
    maximum_trajectory_difference = float(np.max(ddim_difference))
    maximum_final_difference = float(np.max(np.abs(ddim_a[-1] - ddim_b[-1])))
    centers = np.asarray(CENTERS)
    final_samples = [target, ddim_a[-1], ddim_b[-1], ddpm_a[-1], ddpm_b[-1]]
    final_names = ["Exact target", "DDIM A", "DDIM B", "DDPM A", "DDPM B"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.trajectory_figure.parent.mkdir(parents=True, exist_ok=True)
    args.samples_figure.parent.mkdir(parents=True, exist_ok=True)
    coordinate_limit = plot_sample_trajectories(paths, path_names, args.trajectory_figure)
    plot_final_samples(final_samples, final_names, centers, args.samples_figure)
    np.savez_compressed(
        args.out_dir / "trajectories-and-samples.npz",
        initial_gaussian_samples=np.asarray(initial),
        target_samples=target,
        ddim_trajectory_a=ddim_a,
        ddim_trajectory_b=ddim_b,
        ddpm_trajectory_noise_a=ddpm_a,
        ddpm_trajectory_noise_b=ddpm_b,
        ddpm_noise_draw_a=np.asarray(ddpm_noise_a),
        ddpm_noise_draw_b=np.asarray(ddpm_noise_b),
        betas=np.asarray(betas),
        ddim_levels=np.asarray(levels),
    )
    settings = {
        "seed": args.seed,
        "num_samples_per_run": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": checkpoint_update,
        "same_initial_gaussian_samples_for_all_runs": True,
        "ddim_runs": 2,
        "ddim_levels": "all 1000 levels, 1000 down to 1",
        "ddim_reverse_noise": "none; deterministic eta=0",
        "ddpm_runs": 2,
        "ddpm_reverse_noise": "independent draws from different split PRNG keys",
        "largest_absolute_ddim_trajectory_difference": maximum_trajectory_difference,
        "largest_absolute_ddim_final_sample_difference": maximum_final_difference,
        "coordinate_limit_for_trajectory_figure": coordinate_limit,
        "trajectory_figure": str(args.trajectory_figure),
        "final_samples_figure": str(args.samples_figure),
        "jax_backend": jax.default_backend(),
    }
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    shutil.copy2(args.config, args.out_dir / "diffusion.toml")

    print(f"DDIM levels: {NUM_STEPS}; samples per run: {NUM_SAMPLES}")
    print(f"Largest absolute difference between DDIM trajectories: {maximum_trajectory_difference:.9g}")
    print(f"Largest absolute difference between DDIM final samples: {maximum_final_difference:.9g}")
    print(f"Saved trajectory plot to {args.trajectory_figure}")
    print(f"Saved final sample comparison to {args.samples_figure}")
    print(f"Saved raw arrays and settings to {args.out_dir}")


if __name__ == "__main__":
    main()
