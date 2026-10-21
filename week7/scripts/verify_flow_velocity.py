#!/usr/bin/env python3
"""Compare the trained flow velocity field with the exact mixture field."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffusion import sample_target
from flow_matching import exact_mixture_velocity, predict_velocity
from sampling_metrics import relative_rms_error

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def load_checkpoint(path: Path, num_layers: int) -> tuple:
    with np.load(path) as saved:
        params = tuple((jnp.asarray(saved[f"layer_{i}_weights"]),
                        jnp.asarray(saved[f"layer_{i}_biases"]))
                       for i in range(num_layers))
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("flow.toml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("artifacts/flow-training/checkpoint-update-25000.npz"))
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("evidence/flow-velocity-fields.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/flow-velocity"))
    args = parser.parse_args()
    if args.grid_size < 3:
        parser.error("grid size must be at least 3")
    config = tomllib.loads(args.config.read_text())
    model = config["model"]
    with np.load(args.checkpoint) as saved:
        checkpoint_update = int(saved["update"])
    params = load_checkpoint(args.checkpoint, len(model["hidden_widths"]) + 1)

    coordinates = np.linspace(-4.5, 4.5, args.grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = jnp.asarray(np.stack((xx.ravel(), yy.ravel()), axis=-1))
    settings = {
        "grid_size": args.grid_size,
        "grid_range": [-4.5, 4.5],
        "metric_sample_size": 4000,
        "metric_distribution": "x_s=(1-s)*x0+s*z with x0~target and z~N(0,I)",
        "evaluated_s": [0.1, 0.5, 0.9],
        "checkpoint": str(args.checkpoint),
        "checkpoint_update": checkpoint_update,
        "relative_rms_definition": "sqrt(mean((pred-exact)^2))/sqrt(mean(exact^2))",
        "jax_backend": jax.default_backend(),
    }
    results = {}
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    for row, s in enumerate((0.1, 0.5, 0.9)):
        exact = np.asarray(exact_mixture_velocity(points, s))
        grid_times = jnp.full((len(points),), s, dtype=jnp.float32)
        learned_grid = np.asarray(predict_velocity(
            params, points, grid_times,
            int(model["time_features"]),
            float(model["time_frequency_base"]),
            float(model["time_scale"]),
        ))
        eval_key = jax.random.fold_in(jax.random.PRNGKey(8434), row)
        data_key, noise_key = jax.random.split(eval_key)
        x0 = sample_target(data_key, 4000)
        noise = jax.random.normal(noise_key, x0.shape, dtype=x0.dtype)
        evaluation_samples = (1.0 - s) * x0 + s * noise
        eval_times = jnp.full((len(evaluation_samples),), s, dtype=jnp.float32)
        learned_eval = np.asarray(predict_velocity(
            params, evaluation_samples, eval_times,
            int(model["time_features"]),
            float(model["time_frequency_base"]),
            float(model["time_scale"]),
        ))
        exact_eval = np.asarray(exact_mixture_velocity(evaluation_samples, s))
        error = relative_rms_error(learned_eval, exact_eval)
        results[str(s)] = {
            "relative_rms_error_on_marginal_samples": error,
            "evaluation_sample_count": len(evaluation_samples),
            "meets_0.1_threshold": error < 0.1,
        }
        for col, (field, title, color) in enumerate((
            (exact, "Exact marginal velocity", "tab:blue"),
            (learned_grid, "Learned velocity", "tab:orange"),
        )):
            ax = axes[row, col]
            ax.scatter(np.asarray(evaluation_samples)[::4, 0],
                       np.asarray(evaluation_samples)[::4, 1],
                       s=3, color="gray", alpha=0.12, linewidths=0, zorder=0)
            ax.quiver(xx, yy, field[:, 0].reshape(xx.shape), field[:, 1].reshape(yy.shape),
                      color=color, angles="xy", scale_units="xy", scale=5.0, width=0.003)
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.grid(alpha=0.18)
            ax.set_title(f"s={s:g}: {title}" + (f" (relative RMS {error:.4f})" if col else ""))
    fig.suptitle("Flow-matching velocity fields on a spatial grid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    settings["figure"] = str(args.output)
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.out_dir / "relative-rms.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved velocity fields to {args.output}")
    for s, row in results.items():
        print(f"s={s}: relative RMS={row['relative_rms_error_on_marginal_samples']:.6f}")


if __name__ == "__main__":
    main()
