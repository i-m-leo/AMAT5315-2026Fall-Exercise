#!/usr/bin/env python3
"""Compare frozen CD-1/CD-10 RBMs from matched digit and random starts."""
from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rbm import gibbs_steps
from plot_rbm import spin_figure


def load_test_images(directory: Path, count: int) -> np.ndarray:
    with gzip.open(directory / "t10k-images-idx3-ubyte.gz", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or count > n:
            raise ValueError("invalid MNIST test image IDX file or count")
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return (pixels[:count * rows * cols].reshape(count, rows * cols) >= 128).astype(np.uint8)


def model_params(path: Path):
    with np.load(path) as z:
        return tuple(jnp.asarray(z[k], dtype=jnp.float32)
                     for k in ("weights", "visible_bias", "hidden_bias"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("inputs/mnist"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/rbm"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--test-images", type=int, default=8,
                        help="number of held-out starting digits (each gets configured chains)")
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--seed", type=int, default=6607)
    args = parser.parse_args()
    targets = sorted(set(args.steps))
    if not targets or targets[0] < 1:
        raise ValueError("sampling steps must be positive")
    if args.chains < 1:
        raise ValueError("chains must be positive")

    digits = load_test_images(args.data, args.test_images)
    n_visible = digits.shape[1]
    key = jax.random.PRNGKey(args.seed)
    # Uniform independent Bernoulli(1/2) visible configurations, used as spin
    # starts after conversion v=(s+1)/2. Identical starts/keys for both RBMs.
    random_starts = np.asarray(jax.random.bernoulli(
        jax.random.fold_in(key, 0), 0.5, (args.test_images, n_visible)), dtype=np.uint8)
    digit_starts = np.repeat(digits[:, None, :], args.chains, axis=1)
    noise_starts = np.repeat(random_starts[:, None, :], args.chains, axis=1)
    starts = {"heldout": digit_starts, "random": noise_starts}

    saved: dict[str, np.ndarray] = {}
    colors: dict[tuple[str, int, str], np.ndarray] = {}
    for cd in (1, 10):
        ppath = args.artifacts / f"cd{cd}-parameters.npz"
        if not ppath.exists():
            raise FileNotFoundError(f"missing frozen CD-{cd} parameters: {ppath}; train first")
        params = model_params(ppath)
        for kind, initial in starts.items():
            # trajectories: [initial-image, chain, checkpoint, visible-pixel]
            state = jnp.asarray(initial.reshape(-1, n_visible), dtype=jnp.float32)
            checkpoints = [np.asarray(state, dtype=np.uint8).reshape(*initial.shape)]
            current_step = 0
            for target in targets:
                state = gibbs_steps(jax.random.fold_in(key, target),
                                    state, params, target - current_step)
                current_step = target
                checkpoints.append(np.asarray(state, dtype=np.uint8).reshape(*initial.shape))
            # First checkpoint is start; later checkpoints represent target steps.
            arr = np.stack(checkpoints, axis=2)
            saved[f"cd{cd}_{kind}"] = arr
            colors[(kind, cd, "all")] = arr
    args.artifacts.mkdir(parents=True, exist_ok=True)
    chain_file = args.artifacts / "sampling-chains.npz"
    np.savez_compressed(chain_file, **saved)
    metadata = {"seed": args.seed, "starts": list(starts), "start_count": args.test_images,
                "chains_per_start": args.chains, "steps": [0, *targets],
                "matching": "identical binarized held-out starts, random starts, and Gibbs keys across frozen CD-1/CD-10 models",
                "random_start_distribution": "independent Bernoulli(0.5) visible units (equivalently uniform +/-1 spins)"}
    (args.artifacts / "sampling-settings.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Show first chain for four initial states in each start family and both models.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in starts:
        fig, axes = plt.subplots(2, 5, figsize=(12, 5.5), layout="constrained")
        for row, cd in enumerate((1, 10)):
            arr = saved[f"cd{cd}_{kind}"]
            for col in range(5):
                ax = axes[row, col]
                ax.imshow(2 * arr[0, 0, col].reshape(28, 28).astype(np.int8) - 1,
                          cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
                ax.set_title(f"{[0, *targets][col]} steps")
                ax.set_axis_off()
            axes[row, 0].set_ylabel(f"CD-{cd}", rotation=0, labelpad=35, va="center")
        fig.suptitle(f"MNIST {kind} starts · matched chains (row = frozen model)")
        path = args.evidence / f"rbm-{kind}-sampling.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170)
        plt.close(fig)

    # Held-out reconstructions are juxtaposed with input and label when labels exist.
    label_path = args.data / "t10k-labels-idx1-ubyte.gz"
    with gzip.open(label_path, "rb") as f:
        magic, nlabels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)[:args.test_images]
    inputs = digit_starts[:, 0]
    fig, axes = plt.subplots(args.test_images, 3, figsize=(7, args.test_images * 1.5),
                             squeeze=False, layout="constrained")
    for i in range(args.test_images):
        for col, (im, title) in enumerate(((inputs[i], f"Input label {int(labels[i])}"),
                                            (saved["cd1_heldout"][i, 0, -1], "CD-1 final"),
                                            (saved["cd10_heldout"][i, 0, -1], "CD-10 final"))):
            axes[i, col].imshow(im.reshape(28, 28), cmap="gray", vmin=0, vmax=1,
                                interpolation="nearest")
            axes[i, col].set_title(title if i == 0 else "")
            axes[i, col].set_axis_off()
    fig.suptitle("Held-out digit reconstruction after 1000 Gibbs steps")
    fig.savefig(args.evidence / "rbm-heldout-reconstructions.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"chains": str(chain_file), "settings": metadata,
                      "figures": ["rbm-heldout-sampling.png", "rbm-random-sampling.png",
                                  "rbm-heldout-reconstructions.png"]}, indent=2))


if __name__ == "__main__":
    main()
