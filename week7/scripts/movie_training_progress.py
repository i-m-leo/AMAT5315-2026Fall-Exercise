#!/usr/bin/env python3
"""Animate matched DDPM paths and noise-prediction loss across checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from diffusion import beta_schedule, ddpm_trajectory

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

NUM_STEPS = 1000
NUM_PATHS = 10


def load_checkpoint(path: Path, expected_update: int, num_layers: int) -> tuple:
    with np.load(path) as saved:
        update = int(saved["update"])
        params = tuple((jnp.asarray(saved[f"layer_{i}_weights"]),
                        jnp.asarray(saved[f"layer_{i}_biases"]))
                       for i in range(num_layers))
    if update != expected_update:
        raise ValueError(f"{path} stores update {update}, expected {expected_update}")
    return params


def load_loss_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    updates = np.asarray([int(row["update"]) for row in rows], dtype=np.int32)
    losses = np.asarray([float(row["noise_mse"]) for row in rows], dtype=np.float32)
    if len(updates) == 0 or not np.all(np.diff(updates) == 1):
        raise ValueError("loss history must have one consecutive row per update, including 0")
    return updates, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/training-progress"))
    parser.add_argument("--output", type=Path, default=Path("evidence/training-progress.gif"))
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--loss-window", type=int, default=250)
    args = parser.parse_args()
    if args.fps < 1 or args.loss_window < 1:
        parser.error("fps and loss-window must be positive")

    config = tomllib.loads(args.config.read_text())
    model_config = config["model"]
    training_config = config["training"]
    checkpoint_updates = tuple(int(value) for value in training_config["checkpoint_updates"])
    history_updates, losses = load_loss_history(args.training_dir / "loss-history.csv")
    if history_updates[-1] < checkpoint_updates[-1]:
        parser.error("loss history does not reach the final configured checkpoint")

    betas = beta_schedule(NUM_STEPS, 0.0001, 0.02)
    key = jax.random.PRNGKey(args.seed)
    initial_key, reverse_key = jax.random.split(key)
    initial_samples = jax.random.normal(initial_key, (NUM_PATHS, 2), dtype=jnp.float32)
    # The rows are aligned with reverse scan order: t=1000,...,1. The final
    # row (t=1) is ignored by DDPM, as required for its deterministic last step.
    reverse_noises = jax.random.normal(
        reverse_key, (NUM_STEPS, NUM_PATHS, 2), dtype=jnp.float32
    )

    sample_paths = []
    jit_trajectory = jax.jit(lambda params: ddpm_trajectory(
        params,
        initial_samples,
        reverse_noises,
        betas,
        int(model_config["time_features"]),
        float(model_config["time_frequency_base"]),
        float(model_config["time_offset"]),
    ))
    for update in checkpoint_updates:
        checkpoint = args.training_dir / f"checkpoint-update-{update:05d}.npz"
        if not checkpoint.is_file():
            parser.error(f"missing checkpoint: {checkpoint}")
        params = load_checkpoint(checkpoint, update, len(model_config["hidden_widths"]) + 1)
        path = np.asarray(jit_trajectory(params))  # x_T,...,x_0, ten matched paths.
        if path.shape != (NUM_STEPS + 1, NUM_PATHS, 2) or not np.isfinite(path).all():
            parser.error(f"invalid or non-finite generation path at update {update}")
        sample_paths.append(path)
        print(f"Generated matched paths with checkpoint {update}", flush=True)
    sample_paths_array = np.asarray(sample_paths, dtype=np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "generation-paths.npz",
        checkpoint_updates=np.asarray(checkpoint_updates, dtype=np.int32),
        initial_samples=np.asarray(initial_samples),
        reverse_noises=np.asarray(reverse_noises),
        paths=sample_paths_array,
        loss_updates=history_updates,
        loss_history=losses,
    )

    # Each stage uses a symmetric view large enough to include all its paths;
    # the current coordinate limit is written on the frame to make rescaling clear.
    stage_limits = [max(6.0, float(np.max(np.abs(paths))) * 1.08)
                    for paths in sample_paths_array]
    maximum_loss = max(1.05, float(np.max(losses)) * 1.05)
    colors = plt.get_cmap("tab10").colors
    fig, (path_ax, loss_ax) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lines = [path_ax.plot([], [], color=colors[i], linewidth=1.0, alpha=0.8,
                          label=f"Path {i + 1}")[0] for i in range(NUM_PATHS)]
    starts = path_ax.scatter(np.asarray(initial_samples)[:, 0],
                             np.asarray(initial_samples)[:, 1],
                             marker="o", s=32, facecolors="none", edgecolors=colors,
                             linewidths=1.2, zorder=3)
    endpoints = path_ax.scatter(np.zeros(NUM_PATHS), np.zeros(NUM_PATHS),
                               marker="*", s=65, c=colors, edgecolors="black",
                               linewidths=0.3, zorder=4)
    path_ax.set_xlim(-6.0, 6.0)
    path_ax.set_ylim(-6.0, 6.0)
    path_ax.set_aspect("equal", adjustable="box")
    path_ax.set_xlabel("x₁")
    path_ax.set_ylabel("x₂")
    path_ax.set_title("DDPM reverse paths (○ xT, ★ x0)")
    view_label = path_ax.text(0.03, 0.97, "", transform=path_ax.transAxes, va="top")
    path_ax.grid(alpha=0.2)
    path_ax.legend(loc="upper right", fontsize=7, ncols=2, framealpha=0.8)

    raw_loss, = loss_ax.plot([], [], color="tab:blue", alpha=0.25, linewidth=0.6,
                             marker=".", markersize=2, label="Minibatch noise MSE")
    average_loss, = loss_ax.plot([], [], color="tab:orange", linewidth=1.8,
                                 marker="o", markersize=3,
                                 label=f"Moving average (window {args.loss_window})")
    current_update = loss_ax.axvline(0, color="black", linestyle="--", linewidth=1)
    loss_ax.set_xlim(0, max(1, checkpoint_updates[-1]))
    loss_ax.set_ylim(0, maximum_loss)
    loss_ax.set_xlabel("Training update")
    loss_ax.set_ylabel("Noise-prediction MSE")
    loss_ax.set_title("Training-loss history so far")
    loss_ax.grid(alpha=0.2)
    loss_ax.legend(fontsize=8)
    stage_label = fig.suptitle("")

    def update_frame(frame_index: int):
        update = checkpoint_updates[frame_index]
        paths = sample_paths_array[frame_index]
        coordinate_limit = stage_limits[frame_index]
        path_ax.set_xlim(-coordinate_limit, coordinate_limit)
        path_ax.set_ylim(-coordinate_limit, coordinate_limit)
        view_label.set_text(f"view ±{coordinate_limit:.1f}")
        for path_index, line in enumerate(lines):
            line.set_data(paths[:, path_index, 0], paths[:, path_index, 1])
        endpoints.set_offsets(paths[-1])

        visible = history_updates <= update
        x_history = history_updates[visible]
        y_history = losses[visible]
        raw_loss.set_data(x_history, y_history)
        window = min(args.loss_window, len(y_history))
        cumulative = np.cumsum(np.insert(y_history, 0, 0.0))
        moving = (cumulative[window:] - cumulative[:-window]) / window
        average_loss.set_data(x_history[window - 1:], moving)
        loss_ax.set_xlim(-1, max(1, update))
        current_update.set_xdata([update, update])
        stage_label.set_text(f"Noise predictor at training update {update}")
        return (*lines, endpoints, raw_loss, average_loss, current_update, stage_label, view_label)

    animation = FuncAnimation(fig, update_frame, frames=len(checkpoint_updates),
                              interval=1000 / args.fps, blit=False, repeat=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    settings = {
        "config": str(args.config),
        "training_dir": str(args.training_dir),
        "checkpoint_updates": list(checkpoint_updates),
        "num_paths": NUM_PATHS,
        "num_diffusion_steps": NUM_STEPS,
        "seed": args.seed,
        "shared_initial_gaussian_samples_across_stages": True,
        "shared_reverse_step_noise_across_stages": True,
        "reverse_step_noise_order": "t=1000 down to t=1; t=1 noise unused",
        "coordinate_limits_by_stage": {
            str(update): [-limit, limit]
            for update, limit in zip(checkpoint_updates, stage_limits)
        },
        "loss_history_visible_through_each_stage_update": True,
        "loss_window": args.loss_window,
        "fps": args.fps,
        "output": str(args.output),
    }
    (args.out_dir / "movie-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved {len(checkpoint_updates)}-stage movie to {args.output}")
    print(f"Saved matched generation paths and settings to {args.out_dir}")


if __name__ == "__main__":
    main()
