#!/usr/bin/env python3
"""Complete held-out MNIST digits by clamped block-Gibbs sampling."""
from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_test_images(directory: Path, count: int) -> np.ndarray:
    with gzip.open(directory / "t10k-images-idx3-ubyte.gz", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or rows != 28 or cols != 28 or not 0 < count <= n:
            raise ValueError("invalid MNIST test image IDX file or count")
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    if pixels.size != n * rows * cols:
        raise ValueError("inconsistent MNIST test image file")
    return (pixels[:count * rows * cols].reshape(count, rows, cols) >= 128).astype(np.uint8)


def load_params(path: Path):
    with np.load(path) as z:
        return tuple(jnp.asarray(z[k], dtype=jnp.float32)
                     for k in ("weights", "visible_bias", "hidden_bias"))


def clamped_gibbs(key, v, observed, mask, params):
    """One binary RBM block-Gibbs transition, keeping observed visibles fixed."""
    w, vb, hb = params
    v = jnp.where(mask, observed, v)
    kh, kv = jax.random.split(key)
    h_prob = jax.nn.sigmoid(v @ w + hb)
    h = jax.random.bernoulli(kh, h_prob).astype(v.dtype)
    v_prob = jax.nn.sigmoid(h @ w.T + vb)
    proposed = jax.random.bernoulli(kv, v_prob).astype(v.dtype)
    return jnp.where(mask, observed, proposed)


def run_chain(key, initial, observed, mask, params, steps):
    def body(i, state):
        return clamped_gibbs(jax.random.fold_in(key, i), state, observed, mask, params)
    return jax.lax.fori_loop(0, steps, body, initial)


def plot_model(states: np.ndarray, observed: np.ndarray, *, model: str,
               steps: list[int], output: Path, display_digits: int, display_chains: int):
    # states shape [digit, chain, checkpoint, 28, 28], checkpoint 0 is start.
    nrows = display_digits * display_chains
    fig, axes = plt.subplots(nrows, len(steps) + 1,
                             figsize=((len(steps) + 1) * 1.6, nrows * 0.62),
                             squeeze=False, layout="constrained")
    titles = ["Random start", *(f"{s} steps" for s in steps)]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9)
    for digit in range(display_digits):
        for chain in range(display_chains):
            row = digit * display_chains + chain
            for col in range(len(steps) + 1):
                ax = axes[row, col]
                # Mark observed upper pixels in gold; generated lower pixels in grayscale.
                ax.imshow(states[digit, chain, col], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest", origin="upper")
                ax.axhline(13.5, color="#ffb000", lw=0.8)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"d{digit} · c{chain}", rotation=0, labelpad=25,
                                  va="center", fontsize=7)
    fig.suptitle(f"MNIST lower-half completion · frozen {model} · gold line = clamp boundary")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("inputs/mnist"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/rbm"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--digits", type=int, default=8)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--display-digits", type=int, default=4)
    parser.add_argument("--display-chains", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10, 100, 1000])
    parser.add_argument("--seed", type=int, default=6608)
    args = parser.parse_args()
    targets = sorted(set(args.steps))
    if not targets or targets[0] < 1 or args.chains < 1 or args.digits < 1:
        raise ValueError("digits, chains, and sampling steps must be positive")
    if args.display_digits > args.digits or args.display_chains > args.chains:
        raise ValueError("display counts may not exceed sampled counts")

    originals = load_test_images(args.data, args.digits)
    n, height, width = originals.shape
    flat = originals.reshape(n, -1)
    visible = jnp.asarray(flat[:, None, :], dtype=jnp.float32)
    observed = jnp.broadcast_to(visible, (n, args.chains, height * width)).reshape(-1, height * width)
    mask2d = np.zeros((height, width), dtype=bool)
    mask2d[:height // 2] = True
    mask = jnp.asarray(np.broadcast_to(mask2d.reshape(1, -1), (n * args.chains, height * width)))

    root = jax.random.PRNGKey(args.seed)
    # Identical Bernoulli(1/2) lower-half starts and keys across CD models.
    noise = jax.random.bernoulli(jax.random.fold_in(root, 0), 0.5,
                                 (n, args.chains, height, width)).astype(jnp.float32)
    initial = np.asarray(noise).reshape(n, args.chains, -1)
    upper = np.broadcast_to(flat[:, None, :], (n, args.chains, height * width)).copy()
    upper[:, :, height // 2 * width:] = initial[:, :, height // 2 * width:]
    initial = upper
    initial[:, :, :height // 2 * width] = np.broadcast_to(
        flat[:, None, :height // 2 * width], (n, args.chains, height // 2 * width))
    observed_flat = np.broadcast_to(flat[:, None, :], initial.shape).reshape(-1, height * width)
    state_initial = jnp.asarray(initial.reshape(-1, height * width), dtype=jnp.float32)

    trajectories = {}
    for cd in (1, 10):
        path = args.artifacts / f"cd{cd}-parameters.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing frozen CD-{cd} parameters: {path}")
        params = load_params(path)
        state = state_initial
        checkpoints = [initial.copy()]
        previous = 0
        for target in targets:
            keys = jax.random.split(jax.random.fold_in(root, target), n * args.chains)
            state = jax.jit(jax.vmap(run_chain, in_axes=(0, 0, 0, 0, None, None)))(
                keys, state, jnp.asarray(observed_flat, dtype=jnp.float32), mask, params,
                target - previous)
            # Explicitly assert the clamp invariant at every saved checkpoint.
            sampled = np.asarray(state, dtype=np.uint8).reshape(n, args.chains, height, width)
            if not np.array_equal(sampled[:, :, :height // 2],
                                  np.broadcast_to(originals[:, None, :height // 2],
                                                  (n, args.chains, height // 2, width))):
                raise AssertionError("observed upper half changed during Gibbs completion")
            checkpoints.append(sampled.reshape(n, args.chains, -1))
            previous = target
        trajectories[f"cd{cd}"] = np.stack(checkpoints, axis=2).reshape(
            n, args.chains, len(checkpoints), height, width)
        plot_model(trajectories[f"cd{cd}"], originals, model=f"CD-{cd}", steps=targets,
                   output=args.evidence / f"mnist-completion-cd{cd}.png",
                   display_digits=args.display_digits, display_chains=args.display_chains)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    archive = args.artifacts / "completion-chains.npz"
    np.savez_compressed(archive, **trajectories, originals=originals)
    settings = {"seed": args.seed, "digits": args.digits, "chains_per_digit": args.chains,
                "steps": [0, *targets], "observed_region": "upper 14 rows clamped at every Gibbs transition",
                "missing_region_initialization": "matched independent Bernoulli(0.5) pixels",
                "matching": "same held-out digits, random lower halves, and Gibbs keys for frozen CD-1/CD-10",
                "parameters_frozen": True}
    (args.artifacts / "completion-settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps({"archive": str(archive), "settings": settings,
                      "figures": ["mnist-completion-cd1.png", "mnist-completion-cd10.png"]}, indent=2))


if __name__ == "__main__":
    main()
