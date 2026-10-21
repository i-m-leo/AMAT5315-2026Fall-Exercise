#!/usr/bin/env python3
"""Complete four masked CIFAR-10 test images with RePaint and the fixed U-Net."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cifar_sampling import linear_betas, repaint_completion
from cifar_unet import load_cifar_unet, make_parameterized_predictor

REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SEEDS = (7201, 7202, 7203, 7204)


def _to_display(image: np.ndarray) -> np.ndarray:
    image = np.transpose(image, (1, 2, 0))
    return np.clip((image + 1.0) / 2.0, 0.0, 1.0)


def _plot_completions(
    truth: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    completions: np.ndarray,
    variances: np.ndarray,
    mse: np.ndarray,
    labels: np.ndarray,
    output: Path,
) -> None:
    rows = len(truth)
    columns = 8
    fig, axes = plt.subplots(rows, columns, figsize=(16, 2.8 * rows), layout="constrained", squeeze=False)
    headings = ("Truth", "Observed", "Mean of 4", "Pixelwise variance", *(f"Seed {seed}" for seed in SEEDS))
    vmax = max(float(np.max(variances)), 1e-8)
    for row in range(rows):
        axes[row, 0].imshow(_to_display(truth[row]), interpolation="nearest")
        observed_view = _to_display(truth[row]).copy()
        observed_view[np.asarray(mask[row, 0] == 0)] = (0.20, 0.20, 0.20)
        axes[row, 1].imshow(observed_view, interpolation="nearest")
        mean_completion = np.mean(completions[row], axis=0)
        axes[row, 2].imshow(_to_display(mean_completion), interpolation="nearest")
        scalar_variance = np.mean(variances[row], axis=0)
        axes[row, 3].imshow(scalar_variance, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
        for seed_index, _seed in enumerate(SEEDS):
            axes[row, 4 + seed_index].imshow(_to_display(completions[row, seed_index]), interpolation="nearest")
            axes[row, 4 + seed_index].set_xlabel(f"hidden MSE {mse[row, seed_index]:.3f}", fontsize=8)
        for col in range(columns):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(headings[col], fontsize=9)
        axes[row, 0].set_ylabel(CLASS_NAMES[int(labels[row])], rotation=0, labelpad=38, va="center")
    fig.suptitle("RePaint with the fixed CIFAR-10 noise predictor · hidden rows/columns 6:26")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("inputs/cifar-model"))
    parser.add_argument("--images", type=Path, default=Path("inputs/images.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cifar-repaint"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    params, config = load_cifar_unet(args.model_dir)
    predictor = make_parameterized_predictor()
    with np.load(args.images) as archive:
        truth = archive["test"].astype(np.float32)
        labels = archive["labels"].astype(int)
    if truth.shape != (4, 3, 32, 32):
        raise ValueError(f"expected four 32x32 RGB test images, got {truth.shape}")

    mask = np.ones_like(truth, dtype=np.float32)
    mask[:, :, 6:26, 6:26] = 0.0
    observed = mask * truth
    betas = linear_betas(1000)
    completions = np.empty((4, 4, 3, 32, 32), dtype=np.float32)
    mse = np.empty((4, 4), dtype=np.float64)
    observed_changes = np.empty((4, 4), dtype=np.float64)
    evaluations = np.empty((4, 4), dtype=np.int32)
    started = time.time()
    for image_index in range(4):
        for seed_index, seed in enumerate(SEEDS):
            completion, count = repaint_completion(
                jax.random.PRNGKey(seed),
                jnp.asarray(observed[image_index : image_index + 1]),
                jnp.asarray(mask[image_index : image_index + 1]),
                betas,
                predictor,
                params,
            )
            completed = np.asarray(completion[0])
            completions[image_index, seed_index] = completed
            hidden = mask[image_index] == 0
            visible = mask[image_index] == 1
            mse[image_index, seed_index] = np.mean(np.square(completed[hidden] - truth[image_index][hidden]))
            observed_changes[image_index, seed_index] = np.max(
                np.abs(completed[visible] - truth[image_index][visible])
            )
            evaluations[image_index, seed_index] = count
            if count != 99:
                raise AssertionError(f"RePaint used {count} evaluations instead of 99")
            if observed_changes[image_index, seed_index] >= 1e-5:
                raise AssertionError("RePaint changed a visible pixel by at least 1e-5")
            print(
                f"image {image_index} ({CLASS_NAMES[int(labels[image_index])]}), seed {seed}: "
                f"hidden MSE={mse[image_index, seed_index]:.5f}, "
                f"max visible change={observed_changes[image_index, seed_index]:.2g}"
            )
    elapsed = time.time() - started
    variances = np.var(completions, axis=1)
    np.savez_compressed(
        args.out / "inpainting.npz",
        truth=truth,
        observed=observed,
        mask=mask,
        completions=completions,
        pixelwise_variance=variances,
        hidden_pixel_mse=mse,
        observed_max_change=observed_changes,
        seeds=np.asarray(SEEDS, dtype=np.int64),
        labels=labels,
    )
    metrics = {
        "checkpoint": "google/ddpm-cifar10-32",
        "revision": REVISION,
        "steps": 1000,
        "levels": list(range(1000, 0, -20)) + [0],
        "attempts_per_jump_except_final": 2,
        "evaluations_per_completion": 99,
        "hidden_square": "rows and columns 6:26, all RGB channels",
        "seed_per_completion": list(SEEDS),
        "images": [
            {
                "index": image_index,
                "label": CLASS_NAMES[int(labels[image_index])],
                "hidden_pixel_mse_by_seed": {
                    str(seed): float(mse[image_index, seed_index])
                    for seed_index, seed in enumerate(SEEDS)
                },
                "max_observed_pixel_change_by_seed": {
                    str(seed): float(observed_changes[image_index, seed_index])
                    for seed_index, seed in enumerate(SEEDS)
                },
                "evaluations_by_seed": {
                    str(seed): int(evaluations[image_index, seed_index])
                    for seed_index, seed in enumerate(SEEDS)
                },
                "mean_hidden_pixel_mse": float(np.mean(mse[image_index])),
            }
            for image_index in range(4)
        ],
        "overall_mean_hidden_pixel_mse": float(np.mean(mse)),
        "largest_observed_pixel_change": float(np.max(observed_changes)),
        "elapsed_seconds": elapsed,
        "device": jax.default_backend(),
        "model_config": config,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    settings = {
        "checkpoint": "google/ddpm-cifar10-32",
        "revision": REVISION,
        "diffusers_reference": "v0.40.0 UNet2DModel",
        "image_model_index": "mathematical_level - 1",
        "sampler": "RePaint",
        "beta_schedule": "linear(0.0001, 0.02)",
        "sampling_clip": "only Tweedie x0 estimate clipped to [-1, 1]",
        "device": jax.default_backend(),
        "elapsed_seconds": elapsed,
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    figure = args.evidence / "cifar-repaint-inpainting.png"
    _plot_completions(truth, observed, mask, completions, variances, mse, labels, figure)
    print(f"Overall mean hidden-pixel MSE: {np.mean(mse):.5f}")
    print(f"Largest visible-pixel change: {np.max(observed_changes):.2g}")
    print(f"Saved RePaint evidence to {figure}")


if __name__ == "__main__":
    main()
