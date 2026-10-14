#!/usr/bin/env python3
"""Compare lower-half ink overlap for correct and wrong-upper completions."""
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

from complete_rbm import load_params, run_chain


def load_test(directory: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(directory / "t10k-images-idx3-ubyte.gz", "rb") as f:
        magic, n, height, width = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or (height, width) != (28, 28) or count > n:
            raise ValueError("invalid MNIST test image IDX")
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(directory / "t10k-labels-idx1-ubyte.gz", "rb") as f:
        magic_l, n_l = struct.unpack(">II", f.read(8))
        if magic_l != 2049 or n_l != n:
            raise ValueError("invalid MNIST test label IDX")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    images = (raw.reshape(n, height, width) >= 128).astype(np.uint8)
    return images[:count], labels


def select_wrong_indices(labels: np.ndarray, count: int, pool: int) -> np.ndarray:
    """Next different-label test example per target, wrapping in first pool entries."""
    if pool > len(labels) or count > pool:
        raise ValueError("wrong-upper selection pool does not cover requested images")
    result = np.empty(count, dtype=int)
    for i in range(count):
        for offset in range(1, pool + 1):
            candidate = (i + offset) % pool
            if labels[candidate] != labels[i]:
                result[i] = candidate
                break
        else:
            raise ValueError("no different-label donor found")
    return result


def ink_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sørensen-Dice ink overlap, with score 1 when both masks have no ink."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    intersection = np.sum(a & b, axis=-1)
    total = np.sum(a, axis=-1) + np.sum(b, axis=-1)
    return np.where(total == 0, 1.0, 2.0 * intersection / np.maximum(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("inputs/mnist"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/rbm"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    args = parser.parse_args()
    completion_path = args.artifacts / "completion-chains.npz"
    settings = json.loads((args.artifacts / "completion-settings.json").read_text())
    steps = list(map(int, settings["steps"]))
    targets = steps[1:]
    n = int(settings["digits"])
    chains = int(settings["chains_per_digit"])
    originals, labels = load_test(args.data, max(64, n))
    target_digits = originals[:n]
    wrong_ids = select_wrong_indices(labels, n, 64)
    wrong_upper_images = originals[wrong_ids]

    with np.load(completion_path) as archive:
        correct = {f"cd{k}": archive[f"cd{k}"].copy() for k in (1, 10)}
    mask2d = np.zeros((28, 28), dtype=bool)
    mask2d[:14] = True
    mask = jnp.asarray(np.broadcast_to(mask2d.reshape(1, -1), (n * chains, 784)))
    true_flat = target_digits.reshape(n, 784)
    # Clamp donor upper halves, but retain each target's true lower half for scoring.
    wrong_observed = np.broadcast_to(true_flat[:, None, :], (n, chains, 784)).copy()
    wrong_observed[:, :, :14 * 28] = np.broadcast_to(
        wrong_upper_images[:, None, :14].reshape(n, 1, -1), (n, chains, 14 * 28))
    observed_flat = jnp.asarray(wrong_observed.reshape(-1, 784), dtype=jnp.float32)
    initial = correct["cd1"][:, :, 0].reshape(n * chains, 784)
    # Lower halves originate from the same random starts used by the correct-upper runs.
    initial = initial.copy()
    initial[:, :14 * 28] = wrong_observed.reshape(-1, 784)[:, :14 * 28]
    state0 = jnp.asarray(initial, dtype=jnp.float32)
    root = jax.random.PRNGKey(int(settings["seed"]))
    mask_all = mask
    wrong: dict[str, np.ndarray] = {}
    for cd in (1, 10):
        params = load_params(args.artifacts / f"cd{cd}-parameters.npz")
        state = state0
        saved = [initial.reshape(n, chains, 784).reshape(n, chains, 28, 28).copy()]
        previous = 0
        for target in targets:
            keys = jax.random.split(jax.random.fold_in(root, target), n * chains)
            state = jax.jit(jax.vmap(run_chain, in_axes=(0, 0, 0, 0, None, None)))(
                keys, state, observed_flat, mask_all, params, target - previous)
            sample = np.asarray(state, dtype=np.uint8).reshape(n, chains, 28, 28)
            if not np.array_equal(sample[:, :, :14],
                                  np.broadcast_to(wrong_upper_images[:, None, :14],
                                                  (n, chains, 14, 28))):
                raise AssertionError("wrong upper-half observations were not clamped")
            saved.append(sample)
            previous = target
        wrong[f"cd{cd}"] = np.stack(saved, axis=2)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifacts / "wrong-upper-completion-chains.npz", **wrong,
                        donor_indices=wrong_ids, donor_labels=labels[wrong_ids],
                        target_labels=labels[:n])

    # Score only missing lower-half pixels; aggregate chains per digit, then
    # calculate mean and standard error across digits for uncertainty bars.
    truth_lower = target_digits[:, None, 14:, :].astype(bool)
    series = {}
    for condition, data in (("correct upper", correct), ("wrong upper", wrong)):
        for cd in (1, 10):
            digit_means = []
            arr = data[f"cd{cd}"]
            per_step = []
            for checkpoint in range(len(steps)):
                generated = arr[:, :, checkpoint, 14:, :].astype(bool)
                actual = np.broadcast_to(truth_lower, generated.shape)
                scores = ink_overlap(generated.reshape(n, chains, -1),
                                     actual.reshape(n, chains, -1))
                per_step.append(scores)
                digit_means.append(scores.mean(axis=1))
            series[(condition, cd)] = np.stack(digit_means, axis=1)

    metrics = {"steps": steps, "overlap": "2|A∩B|/(|A|+|B|) over lower-half ink pixels; empty/empty=1",
               "wrong_upper_donor_indices": wrong_ids.tolist(),
               "wrong_upper_donor_labels": labels[wrong_ids].astype(int).tolist(),
               "target_labels": labels[:n].astype(int).tolist(), "results": {}}
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    colors = {("correct upper", 1): "#1f77b4", ("correct upper", 10): "#d95f02",
              ("wrong upper", 1): "#6baed6", ("wrong upper", 10): "#fd8d3c"}
    for (condition, cd), values in series.items():
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]) if values.shape[0] > 1 else np.zeros_like(mean)
        label = f"CD-{cd}, {condition}"
        ax.errorbar(steps, mean, yerr=se, marker="o", capsize=3, lw=1.7,
                    color=colors[(condition, cd)], label=label)
        metrics["results"][label] = {"mean": mean.tolist(), "standard_error_across_digits": se.tolist()}
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Gibbs steps")
    ax.set_ylabel("Lower-half ink overlap (Dice score)")
    ax.set_title("Held-out MNIST completion: correct vs wrong upper context")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    args.evidence.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.evidence / "mnist-completion-ink-overlap.png", dpi=180)
    plt.close(fig)
    (args.artifacts / "completion-overlap-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"figure": "evidence/mnist-completion-ink-overlap.png",
                      "metrics": str(args.artifacts / "completion-overlap-metrics.json"),
                      "wrong_donors": wrong_ids.tolist()}, indent=2))


if __name__ == "__main__":
    main()
