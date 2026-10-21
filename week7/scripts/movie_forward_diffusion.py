#!/usr/bin/env python3
"""Animate one fixed sample batch through the recorded forward trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path,
                        default=Path("artifacts/forward-diffusion/trajectory.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("evidence/forward-diffusion-modes.gif"))
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="show every Nth diffusion level (the final level is always included)")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--limit", type=float, default=6.0,
                        help="fixed symmetric coordinate range for both axes")
    args = parser.parse_args()
    if args.frame_stride < 1 or args.fps < 1 or args.limit <= 0:
        parser.error("frame stride, fps, and coordinate limit must be positive")
    if not args.trajectory.is_file():
        parser.error(f"trajectory not found: {args.trajectory}; run scripts/forward_diffusion.py first")

    with np.load(args.trajectory) as raw:
        trajectory = raw["trajectory"]
        betas = raw["betas"]
    if trajectory.ndim != 3 or trajectory.shape[0] != len(betas) + 1 or trajectory.shape[2] != 2:
        parser.error("expected trajectory with shape (T+1, samples, 2) and matching betas")
    if not np.isfinite(trajectory).all():
        parser.error("trajectory contains non-finite coordinates")

    num_steps = trajectory.shape[0] - 1
    frames = np.arange(0, num_steps + 1, args.frame_stride, dtype=int)
    if frames[-1] != num_steps:
        frames = np.append(frames, num_steps)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    points = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 1],
                        s=8, alpha=0.5, linewidths=0)
    label = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
    ax.set_xlim(-args.limit, args.limit)
    ax.set_ylim(-args.limit, args.limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Forward diffusion: Gaussian-mixture modes merge")
    ax.grid(alpha=0.2)

    def update(level: int):
        points.set_offsets(trajectory[level])
        label.set_text(f"t = {level} / {num_steps}")
        return points, label

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps,
                              blit=True, repeat=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    final = trajectory[-1]
    means = final.mean(axis=0)
    variances = final.var(axis=0, ddof=0)
    statistics = {
        "diffusion_level": num_steps,
        "num_samples": int(final.shape[0]),
        "variance_convention": "population variance (ddof=0)",
        "x1_mean": float(means[0]),
        "x1_variance": float(variances[0]),
        "x2_mean": float(means[1]),
        "x2_variance": float(variances[1]),
    }
    settings = {
        "trajectory": str(args.trajectory),
        "output": str(args.output),
        "num_steps": num_steps,
        "num_samples": int(trajectory.shape[1]),
        "frame_stride": args.frame_stride,
        "num_frames": int(len(frames)),
        "fps": args.fps,
        "fixed_axis_limits": [-args.limit, args.limit],
        "same_samples_at_every_level": True,
    }
    artifact_dir = Path("artifacts/forward-diffusion")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "movie-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (artifact_dir / "terminal-statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")

    print(f"Saved {len(frames)}-frame movie to {args.output}")
    print(f"At t={num_steps} (population statistics, n={len(final)}):")
    print(f"  x1: mean={means[0]:.6f}, variance={variances[0]:.6f}")
    print(f"  x2: mean={means[1]:.6f}, variance={variances[1]:.6f}")
    print(f"Saved movie settings and terminal statistics to {artifact_dir}")


if __name__ == "__main__":
    main()
