#!/usr/bin/env python3
"""Generate four CIFAR-10 images from the pinned pretrained JAX U-Net."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cifar_sampling import ddpm_sample_with_snapshots, linear_betas
from cifar_unet import load_cifar_unet, make_parameterized_predictor

REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def _gallery_utility():
    """Load Week 6's native-resolution image_gallery function."""
    path = Path(__file__).resolve().parents[2] / "week6" / "scripts" / "visualize.py"
    spec = importlib.util.spec_from_file_location("week6_visualize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Week 6 gallery utility at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image_gallery


def _display_uint8(channel_first: np.ndarray) -> np.ndarray:
    image = np.transpose(channel_first, (0, 2, 3, 1))
    return np.rint(np.clip((image + 1.0) / 2.0, 0.0, 1.0) * 255.0).astype(np.uint8)


def _plot_stages(snapshots: dict[int, np.ndarray], output: Path) -> None:
    levels = (1000, 200, 50, 0)
    titles = {1000: "$x_T$ · Gaussian noise", 200: "$x_{200}$", 50: "$x_{50}$", 0: "$x_0$ · generated"}
    fig, axes = plt.subplots(4, len(levels), figsize=(9, 9), layout="constrained")
    for row in range(4):
        for col, level in enumerate(levels):
            axes[row, col].imshow(_display_uint8(snapshots[level])[row], interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[level])
        axes[row, 0].set_ylabel(f"sample {row + 1}", rotation=0, labelpad=38, va="center")
    fig.suptitle("Pretrained CIFAR-10 predictor · 1000-step DDPM reverse process")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("inputs/cifar-model"))
    parser.add_argument("--images", type=Path, default=Path("inputs/images.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cifar-ddpm"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--seed", type=int, default=7401)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    params, config = load_cifar_unet(args.model_dir)
    predictor = make_parameterized_predictor()
    key = jax.random.PRNGKey(args.seed)
    initial_key, reverse_key = jax.random.split(key)
    initial = jax.random.normal(initial_key, (4, 3, 32, 32), dtype=jnp.float32)
    started = time.time()
    final, snapshots = ddpm_sample_with_snapshots(
        reverse_key,
        initial,
        linear_betas(1000),
        predictor,
        params,
        snapshot_levels=(1000, 200, 50, 0),
    )
    elapsed = time.time() - started
    print(f"Generated four images in {elapsed:.1f}s on {jax.default_backend()}.")

    np.savez_compressed(args.out / "samples.npz", **{f"level_{level}": data for level, data in snapshots.items()})
    settings = {
        "seed": args.seed,
        "steps": 1000,
        "beta_schedule": "linear(0.0001, 0.02)",
        "checkpoint": "google/ddpm-cifar10-32",
        "revision": REVISION,
        "weight_file": "diffusion_pytorch_model.safetensors",
        "diffusers_reference": "v0.40.0 UNet2DModel",
        "model_time_index": "mathematical_level - 1",
        "sampling_clip": False,
        "display_mapping": "clip((x + 1) / 2, 0, 1)",
        "device": jax.default_backend(),
        "elapsed_seconds": elapsed,
        "model_config": config,
    }
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    stages_path = args.evidence / "cifar-ddpm-reverse-process.png"
    _plot_stages(snapshots, stages_path)

    with np.load(args.images) as archive:
        real = archive["test"].astype(np.float32)
        labels = archive["labels"].astype(int)
    gallery = _gallery_utility()
    generated_hwc = _display_uint8(np.asarray(final))
    real_hwc = _display_uint8(real)
    gallery(
        generated_hwc,
        count=4,
        columns=2,
        title="Unconditional generated CIFAR-10 samples",
        output=args.evidence / "cifar-generated-gallery.png",
    )
    gallery(
        real_hwc,
        labels,
        class_names=CLASS_NAMES,
        count=4,
        columns=2,
        title="Real CIFAR-10 test images",
        output=args.evidence / "cifar-real-gallery.png",
    )
    print(f"Saved DDPM stages to {stages_path}")
    print("Saved generated and real-image galleries using week6/scripts/visualize.py.")


if __name__ == "__main__":
    main()
