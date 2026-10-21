#!/usr/bin/env python3
"""Compare DDPM, DDIM, and flow matching across sampling step counts."""
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
    ddpm_subsample_trajectory,
    predict_noise,
    sample_target,
    select_ddim_timesteps,
)
from flow_matching import flow_trajectory, predict_velocity
from mode_metrics import assign_nearest_mode, summarize_modes
from sampling_metrics import energy_distance_unbiased

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

TOTAL_STEPS = 1000
SAMPLE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 1000)
NUM_SAMPLES = 4000
NUM_NULL_PAIRS = 20


def load_checkpoint(path: Path, num_layers: int) -> tuple:
    with np.load(path) as saved:
        return tuple((jnp.asarray(saved[f"layer_{i}_weights"]),
                      jnp.asarray(saved[f"layer_{i}_biases"]))
                     for i in range(num_layers))


def _metrics(samples: np.ndarray, target_samples: np.ndarray,
             centers: np.ndarray) -> dict:
    labels = assign_nearest_mode(samples, centers)
    per_mode = summarize_modes(samples, centers, labels)
    fractions = np.asarray([item["population_fraction"] for item in per_mode])
    widths = np.asarray([item["coordinate_std"] for item in per_mode], dtype=np.float64)
    return {
        "energy_distance": energy_distance_unbiased(samples, target_samples),
        "max_cluster_fraction_error": float(np.max(np.abs(fractions - 0.25))),
        "max_coordinate_width_error": float(np.max(np.abs(widths - 0.30))),
        "max_coordinate_center_error": float(max(
            np.max(np.abs(np.asarray(item["empirical_center"]) - item["reference_center"]))
            for item in per_mode if item["population_count"]
        )),
        "per_mode": per_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--flow-config", type=Path, default=Path("flow.toml"))
    parser.add_argument("--diffusion-checkpoint", type=Path,
                        default=Path("artifacts/training/checkpoint-update-25000.npz"))
    parser.add_argument("--flow-checkpoint", type=Path,
                        default=Path("artifacts/flow-training/checkpoint-update-25000.npz"))
    parser.add_argument("--seed", type=int, default=53150)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sampler-comparison"))
    parser.add_argument("--figure", type=Path,
                        default=Path("evidence/sampler-step-comparison.png"))
    args = parser.parse_args()
    for checkpoint in (args.diffusion_checkpoint, args.flow_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")

    diffusion_cfg = tomllib.loads(args.diffusion_config.read_text())["model"]
    flow_cfg = tomllib.loads(args.flow_config.read_text())["model"]
    diffusion_params = load_checkpoint(args.diffusion_checkpoint,
                                       len(diffusion_cfg["hidden_widths"]) + 1)
    flow_params = load_checkpoint(args.flow_checkpoint, len(flow_cfg["hidden_widths"]) + 1)
    betas = beta_schedule(TOTAL_STEPS, 0.0001, 0.02)
    centers = np.asarray(CENTERS)

    base_key = jax.random.PRNGKey(args.seed)
    initial_key, target_key, runs_key, null_key = jax.random.split(base_key, 4)
    initial = jax.random.normal(initial_key, (NUM_SAMPLES, 2), dtype=jnp.float32)
    target_samples = np.asarray(sample_target(target_key, NUM_SAMPLES))

    def noise_predictor(x_t, level):
        return predict_noise(
            diffusion_params, x_t, level,
            int(diffusion_cfg["time_features"]),
            float(diffusion_cfg["time_frequency_base"]),
            float(diffusion_cfg["time_offset"]),
        )

    def velocity_predictor(x_s, s):
        return predict_velocity(
            flow_params, x_s, s,
            int(flow_cfg["time_features"]),
            float(flow_cfg["time_frequency_base"]),
            float(flow_cfg["time_scale"]),
        )

    @jax.jit
    def sample_ddpm(x_terminal, grid, noise):
        return ddpm_subsample_trajectory(
            diffusion_params, x_terminal, grid, noise, betas,
            int(diffusion_cfg["time_features"]),
            float(diffusion_cfg["time_frequency_base"]),
            float(diffusion_cfg["time_offset"]),
        )[-1]

    @jax.jit
    def sample_ddim(x_terminal, levels):
        return ddim_trajectory(x_terminal, levels, betas, noise_predictor)[-1]

    @jax.jit
    def sample_flow(x_terminal, times):
        return flow_trajectory(x_terminal, times, velocity_predictor)[-1]

    results: dict[str, dict] = {}
    samples_by_run: dict[str, np.ndarray] = {}
    run_index = 0
    for method in ("ddpm", "ddim", "flow"):
        counts = SAMPLE_COUNTS if method != "flow" else SAMPLE_COUNTS[:-1]
        for steps in counts:
            run_key = jax.random.fold_in(runs_key, run_index)
            run_index += 1
            if method == "ddpm":
                grid = jnp.asarray(np.rint(np.linspace(TOTAL_STEPS, 0, steps + 1)),
                                   dtype=jnp.int32)
                noise = jax.random.normal(run_key, (steps, NUM_SAMPLES, 2), dtype=jnp.float32)
                samples = np.asarray(sample_ddpm(initial, grid, noise))
            elif method == "ddim":
                levels = select_ddim_timesteps(steps, TOTAL_STEPS)
                samples = np.asarray(sample_ddim(initial, levels))
            else:
                times = jnp.linspace(1.0, 0.0, steps + 1, dtype=jnp.float32)
                samples = np.asarray(sample_flow(initial, times))
            key = f"{method}_{steps}"
            samples_by_run[key] = samples
            results[key] = {
                "method": method,
                "steps": steps,
                **_metrics(samples, target_samples, centers),
            }
            print(f"{method:4s} {steps:4d} steps: energy={results[key]['energy_distance']:.6f}, "
                  f"max fraction error={results[key]['max_cluster_fraction_error']:.4f}, "
                  f"max width error={results[key]['max_coordinate_width_error']:.4f}", flush=True)

    null_scores = []
    null_pair_samples = []
    for pair_index in range(NUM_NULL_PAIRS):
        pair_key = jax.random.fold_in(null_key, pair_index)
        key_a, key_b = jax.random.split(pair_key)
        exact_a = np.asarray(sample_target(key_a, NUM_SAMPLES))
        exact_b = np.asarray(sample_target(key_b, NUM_SAMPLES))
        null_pair_samples.append(np.stack((exact_a, exact_b)))
        null_scores.append(energy_distance_unbiased(exact_a, exact_b))
    null_scores = np.asarray(null_scores, dtype=np.float64)
    null_95 = float(np.quantile(null_scores, 0.95))
    print(f"20 exact-vs-exact pairs: energy null 95% quantile={null_95:.6f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"samples_{name}": value for name, value in samples_by_run.items()}
    arrays.update({
        "shared_initial_gaussian_samples": np.asarray(initial),
        "exact_target_reference": target_samples,
        "null_pair_samples": np.asarray(null_pair_samples),
        "null_energy_distances": null_scores,
    })
    np.savez_compressed(args.out_dir / "sampler-results.npz", **arrays)
    settings = {
        "seed": args.seed,
        "num_samples_per_run": NUM_SAMPLES,
        "steps": {"ddpm": list(SAMPLE_COUNTS), "ddim": list(SAMPLE_COUNTS),
                  "flow": list(SAMPLE_COUNTS[:-1])},
        "same_initial_gaussian_samples_across_runs": True,
        "target_reference_samples": NUM_SAMPLES,
        "independent_exact_null_pairs": NUM_NULL_PAIRS,
        "samples_per_null_set": NUM_SAMPLES,
        "energy_distance_estimator": "unbiased U-statistic, Euclidean metric",
        "energy_null_95_percentile": null_95,
        "fraction_error_band_2_standard_errors": 2.0 * np.sqrt(0.25 * 0.75 / NUM_SAMPLES),
        "width_error_tolerance": 0.03,
        "ddpm_checkpoint": str(args.diffusion_checkpoint),
        "flow_checkpoint": str(args.flow_checkpoint),
        "jax_backend": jax.default_backend(),
        "figure": str(args.figure),
    }
    (args.out_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.out_dir / "metrics.json").write_text(json.dumps({
        "null_energy_distances": null_scores.tolist(),
        "null_energy_95_percentile": null_95,
        "samplers": results,
    }, indent=2) + "\n")
    shutil.copy2(args.diffusion_config, args.out_dir / "diffusion.toml")
    shutil.copy2(args.flow_config, args.out_dir / "flow.toml")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    palette = {"ddpm": "tab:blue", "ddim": "tab:orange", "flow": "tab:green"}
    names = {"ddpm": "DDPM", "ddim": "DDIM", "flow": "Flow matching"}
    energy_floor = 1e-4
    axes[0].axhspan(energy_floor, max(energy_floor, null_95), color="gray", alpha=0.2,
                    label="Exact-vs-exact 95% null band")
    fraction_band = 2.0 * np.sqrt(0.25 * 0.75 / NUM_SAMPLES)
    axes[1].axhspan(0, fraction_band, color="gray", alpha=0.2,
                    label="2 binomial standard errors")
    axes[2].axhspan(0, 0.03, color="gray", alpha=0.2, label="0.03 tolerance")
    for method in ("ddpm", "ddim", "flow"):
        rows = sorted((value for value in results.values() if value["method"] == method),
                      key=lambda row: row["steps"])
        steps = np.asarray([row["steps"] for row in rows])
        energy = np.maximum([row["energy_distance"] for row in rows], energy_floor)
        fraction_error = [row["max_cluster_fraction_error"] for row in rows]
        width_error = [row["max_coordinate_width_error"] for row in rows]
        for ax, values in zip(axes, (energy, fraction_error, width_error)):
            ax.plot(steps, values, marker="o", linewidth=1.6,
                    color=palette[method], label=names[method])
    axes[0].set_yscale("log")
    axes[0].set_ylim(energy_floor, max(0.05, max(
        max(row["energy_distance"], energy_floor) for row in results.values()
    ) * 1.2))
    axes[0].set_ylabel("Energy distance (negative estimates shown at 10⁻⁴)")
    axes[1].set_ylabel("Largest cluster-fraction error")
    axes[2].set_ylabel("Largest per-coordinate width error")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Network evaluations / sampling steps")
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=8)
    axes[0].set_title("Distribution discrepancy")
    axes[1].set_title("Cluster populations")
    axes[2].set_title("Cluster widths")
    fig.suptitle("Cost and accuracy of DDPM, DDIM, and flow matching")
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    print(f"Saved sampler metrics/settings to {args.out_dir}")
    print(f"Saved step-comparison figure to {args.figure}")


if __name__ == "__main__":
    main()
