#!/usr/bin/env python3
"""Animate 2000 DDPM reverse paths and compare final samples with the target."""
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
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from diffusion import CENTERS, TARGET_VARIANCE, beta_schedule, ddpm_trajectory, sample_target
from mode_metrics import assign_nearest_mode, summarize_modes

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


def _mode_labels(points: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                 ax, *, show_estimates: list[dict] | None = None) -> None:
    colors = plt.get_cmap("tab10").colors
    for mode in range(len(centers)):
        subset = points[labels == mode]
        ax.scatter(subset[:, 0], subset[:, 1], s=8, alpha=0.35,
                   color=colors[mode], linewidths=0, label=f"Mode {mode + 1}")
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=75, c="black",
               linewidths=1.6, label="Reference centres", zorder=4)
    if show_estimates is not None:
        for mode, stats in enumerate(show_estimates):
            if stats["population_count"]:
                mean = stats["empirical_center"]
                ax.scatter(mean[0], mean[1], marker="*", s=95, color=colors[mode],
                           edgecolors="black", linewidths=0.5, zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/training/checkpoint-update-25000.npz"))
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=5315)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/reverse-diffusion"))
    parser.add_argument("--movie", type=Path, default=Path("evidence/ddpm-reverse-process.gif"))
    parser.add_argument("--comparison", type=Path,
                        default=Path("evidence/ddpm-final-comparison.png"))
    args = parser.parse_args()
    if args.num_samples < 1 or args.frame_stride < 1 or args.fps < 1:
        parser.error("sample count, frame stride, and fps must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    config = tomllib.loads(args.config.read_text())
    model_config = config["model"]
    with np.load(args.checkpoint) as checkpoint:
        checkpoint_update = int(checkpoint["update"])
    params = load_checkpoint(args.checkpoint, checkpoint_update,
                             len(model_config["hidden_widths"]) + 1)

    key = jax.random.PRNGKey(args.seed)
    initial_key, noise_key, target_key = jax.random.split(key, 3)
    initial = jax.random.normal(initial_key, (args.num_samples, 2), dtype=jnp.float32)
    reverse_noises = jax.random.normal(
        noise_key, (NUM_STEPS, args.num_samples, 2), dtype=jnp.float32
    )
    target = sample_target(target_key, args.num_samples)
    betas = beta_schedule(NUM_STEPS, 0.0001, 0.02)
    trajectory_fn = jax.jit(lambda model_params: ddpm_trajectory(
        model_params,
        initial,
        reverse_noises,
        betas,
        int(model_config["time_features"]),
        float(model_config["time_frequency_base"]),
        float(model_config["time_offset"]),
    ))
    trajectory = np.asarray(trajectory_fn(params))  # x_T,...,x_0
    target_np = np.asarray(target)
    if trajectory.shape != (NUM_STEPS + 1, args.num_samples, 2):
        raise RuntimeError(f"unexpected reverse trajectory shape {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise RuntimeError("reverse trajectory contains non-finite coordinates")
    final_samples = trajectory[-1]
    centers = np.asarray(CENTERS)
    generated_labels = assign_nearest_mode(final_samples, centers)
    target_labels = assign_nearest_mode(target_np, centers)
    generated_stats = summarize_modes(final_samples, centers, generated_labels)
    target_stats = summarize_modes(target_np, centers, target_labels)

    frame_indices = np.arange(0, NUM_STEPS + 1, args.frame_stride, dtype=np.int32)
    if frame_indices[-1] != NUM_STEPS:
        frame_indices = np.append(frame_indices, NUM_STEPS)
    coordinate_limit = max(6.0, float(np.max(np.abs(trajectory))) * 1.03)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.movie.parent.mkdir(parents=True, exist_ok=True)
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "reverse-trajectory.npz",
        trajectory=trajectory,
        final_samples=final_samples,
        target_samples=target_np,
        initial_gaussian_samples=np.asarray(initial),
        reverse_step_noises=np.asarray(reverse_noises),
        betas=np.asarray(betas),
        frame_indices=frame_indices,
    )
    settings = {
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_diffusion_steps": NUM_STEPS,
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": checkpoint_update,
        "reverse_step_variance": "beta_t",
        "predicted_x0_clipping": False,
        "noise_added_at_final_step": False,
        "target_centers": centers.tolist(),
        "target_covariance": [[TARGET_VARIANCE, 0.0], [0.0, TARGET_VARIANCE]],
        "target_mixture_weights": [0.25] * 4,
        "frame_stride": args.frame_stride,
        "fps": args.fps,
        "fixed_coordinate_limits": [-coordinate_limit, coordinate_limit],
        "mode_assignment": "nearest reference centre",
        "spread_convention": "population variance and standard deviation (ddof=0)",
        "movie": str(args.movie),
        "comparison_figure": str(args.comparison),
        "jax_backend": jax.default_backend(),
    }
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    stats = {
        "mode_assignment": "nearest reference centre",
        "spread_convention": "population variance and standard deviation (ddof=0)",
        "target_reference": target_stats,
        "generated": generated_stats,
    }
    (args.out_dir / "mode-statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    shutil.copy2(args.config, args.out_dir / "diffusion.toml")

    colors = plt.get_cmap("tab10").colors
    fig, (generated_ax, target_ax) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    generated_artist = generated_ax.scatter(
        trajectory[0, :, 0], trajectory[0, :, 1], s=8, alpha=0.4,
        color=colors[0], linewidths=0,
    )
    target_ax.scatter(target_np[:, 0], target_np[:, 1], s=8, alpha=0.35,
                      c=[colors[i] for i in target_labels], linewidths=0)
    target_ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=75, c="black",
                      linewidths=1.6, label="Mixture centres")
    target_ax.set_title("Target mixture")
    target_ax.legend(loc="upper right", fontsize=8)
    for ax in (generated_ax, target_ax):
        ax.set_xlim(-coordinate_limit, coordinate_limit)
        ax.set_ylim(-coordinate_limit, coordinate_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(alpha=0.18)
    generated_ax.set_title("Generated samples")
    frame_label = fig.suptitle("")

    def update_frame(frame_number: int):
        index = int(frame_indices[frame_number])
        points = trajectory[index]
        generated_artist.set_offsets(points)
        generated_artist.set_color([colors[i] for i in generated_labels])
        level = NUM_STEPS - index
        generated_ax.set_title(f"DDPM reverse samples (t = {level})")
        frame_label.set_text("Reverse diffusion: generated samples vs target")
        return generated_artist, frame_label

    animation = FuncAnimation(fig, update_frame, frames=len(frame_indices),
                              interval=1000 / args.fps, blit=False, repeat=True)
    animation.save(args.movie, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    # Show the terminal comparison and a compact per-mode generated-sample summary.
    fig = plt.figure(figsize=(15, 5.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.5))
    target_ax = fig.add_subplot(grid[0, 0])
    generated_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[0, 2])
    _mode_labels(target_np, target_labels, centers, target_ax)
    _mode_labels(final_samples, generated_labels, centers, generated_ax,
                 show_estimates=generated_stats)
    for ax in (target_ax, generated_ax):
        ax.set_xlim(-coordinate_limit, coordinate_limit)
        ax.set_ylim(-coordinate_limit, coordinate_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(alpha=0.18)
    target_ax.set_title("Target samples")
    generated_ax.set_title(f"DDPM samples (update {checkpoint_update})")
    generated_ax.legend(loc="upper right", fontsize=7, ncols=2)

    table_ax.axis("off")
    table_rows = []
    for item in generated_stats:
        center = item["empirical_center"]
        spread = item["coordinate_std"]
        table_rows.append([
            f"{item['mode_index'] + 1}: {tuple(item['reference_center'])}",
            f"{item['population_count']} ({100 * item['population_fraction']:.1f}%)",
            f"({center[0]:.2f}, {center[1]:.2f})" if center[0] is not None else "empty",
            f"({spread[0]:.2f}, {spread[1]:.2f})" if spread[0] is not None else "empty",
        ])
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Mode (ref.)", "Population n (%)", "Centre (x₁, x₂)", "SD (x₁, x₂)"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.27, 0.23, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    table_ax.set_title("Generated mode statistics\nnearest-centre assignment; SD is population SD")
    fig.suptitle("Final DDPM distribution compared with the target")
    fig.savefig(args.comparison, dpi=180)
    plt.close(fig)

    print(f"Saved reverse-process movie to {args.movie}")
    print(f"Saved final target comparison to {args.comparison}")
    print(f"Saved raw trajectory, settings, and mode statistics to {args.out_dir}")
    print("Per-mode summaries (generated; population SD, ddof=0):")
    for item in generated_stats:
        print(
            f"  mode {item['mode_index'] + 1} ref={item['reference_center']}: "
            f"n={item['population_count']} ({100 * item['population_fraction']:.1f}%), "
            f"centre={item['empirical_center']}, SD={item['coordinate_std']}"
        )


if __name__ == "__main__":
    main()
