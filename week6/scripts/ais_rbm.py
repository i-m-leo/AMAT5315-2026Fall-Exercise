#!/usr/bin/env python3
"""Estimate held-out MNIST RBM NLL by repeated annealed importance sampling."""
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


def load_test(directory: Path, count: int) -> np.ndarray:
    with gzip.open(directory / "t10k-images-idx3-ubyte.gz", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051 or count > n:
            raise ValueError("invalid MNIST test image IDX or requested count")
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return (raw[:count * rows * cols].reshape(count, rows * cols) >= 128).astype(np.float32)


def softplus(x):
    return jnp.logaddexp(jnp.asarray(0, dtype=x.dtype), x)


def ais_log_weights(key, w, vb, hb, particles: int, betas):
    """AIS for joint RBM path with coupling beta*W and fixed biases.

    The beta=0 base is an exactly normalized independent Bernoulli model;
    annealing weights accumulate the interaction term v^T W h.
    """
    kv, kh, kscan = jax.random.split(key, 3)
    pv = jax.nn.sigmoid(vb)
    ph = jax.nn.sigmoid(hb)
    v = jax.random.bernoulli(kv, pv, (particles, vb.shape[0])).astype(jnp.float32)
    h = jax.random.bernoulli(kh, ph, (particles, hb.shape[0])).astype(jnp.float32)
    initial = (kscan, v, h, jnp.zeros((particles,), dtype=jnp.float32))

    # Betas include beta_1..beta_K; incremental deltas are uniform and sum to 1.
    def body(state, beta_next):
        key0, v0, h0, logw, beta_prev = state
        key0, k_h, k_v = jax.random.split(key0, 3)
        interaction = jnp.sum((v0 @ w) * h0, axis=1)
        logw = logw + (beta_next - beta_prev) * interaction
        hprob = jax.nn.sigmoid((v0 @ w) * beta_next + hb)
        h1 = jax.random.bernoulli(k_h, hprob).astype(jnp.float32)
        vprob = jax.nn.sigmoid((h1 @ w.T) * beta_next + vb)
        v1 = jax.random.bernoulli(k_v, vprob).astype(jnp.float32)
        return (key0, v1, h1, logw, beta_next), None

    final, _ = jax.lax.scan(body, (*initial, jnp.array(0., dtype=jnp.float32)), betas)
    return final[3]


@jax.jit
def free_energy(v, w, vb, hb):
    return -(jnp.sum(v * vb, axis=-1) + jnp.sum(softplus(v @ w + hb), axis=-1))


def logmeanexp(x):
    return jax.scipy.special.logsumexp(x) - jnp.log(x.size)


def estimate_model(data, params_path: Path, epochs: list[int], budgets: list[int],
                   repeats: int, anneal_steps: int, seed: int):
    estimates = {}
    betas = jnp.linspace(0., 1., anneal_steps + 1, dtype=jnp.float32)[1:]
    for epoch in epochs:
        prefix = params_path.name.removesuffix("-parameters.npz")
        with np.load(params_path.parent / f"{prefix}-epoch{epoch:02d}.npz") as z:
            w, vb, hb = (jnp.asarray(z[k], dtype=jnp.float32)
                         for k in ("weights", "visible_bias", "hidden_bias"))
        nll_data = float(jnp.mean(free_energy(jnp.asarray(data), w, vb, hb)))
        logz0 = float(jnp.sum(softplus(vb)) + jnp.sum(softplus(hb)))
        max_particles = max(budgets)
        runs = {budget: [] for budget in budgets}
        # Identical base-sample/random streams across models at matching repeat/budget.
        for repeat in range(repeats):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), repeat)
            logw = ais_log_weights(key, w, vb, hb, max_particles, betas)
            for budget in budgets:
                logz = logz0 + float(logmeanexp(logw[:budget]))
                runs[budget].append({"log_z": logz, "nll": nll_data + logz})
        estimates[str(epoch)] = {"mean_free_energy": nll_data, "log_z_base": logz0,
                            "budgets": {str(b): {"runs": runs[b],
                                "mean_nll": float(np.mean([x["nll"] for x in runs[b]])),
                                "std_nll": float(np.std([x["nll"] for x in runs[b]], ddof=1))
                                    if repeats > 1 else 0.0}
                                for b in budgets}}
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("inputs/mnist"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/rbm"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/rbm/metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/rbm/ais-nll.json"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence/rbm-ais-nll.png"))
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--anneal-steps", type=int, default=200)
    parser.add_argument("--test-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6609)
    args = parser.parse_args()
    if args.repeats < 2 or args.anneal_steps < 1 or min(args.budgets) < 1:
        raise ValueError("need at least two repeats and positive budgets/annealing steps")
    data = load_test(args.data, args.test_images)
    estimates = {}
    for cd in (1, 10):
        estimates[f"CD-{cd}"] = estimate_model(
            data, args.artifacts / f"cd{cd}-parameters.npz", args.epochs,
            sorted(set(args.budgets)), args.repeats, args.anneal_steps, args.seed)
    result = {"method": "joint RBM AIS; beta scales W from independent Bernoulli base, biases fixed",
              "nll": "mean held-out visible free energy + estimated log partition function",
              "test_images": args.test_images, "epochs": args.epochs,
              "budgets": sorted(set(args.budgets)), "repeats": args.repeats,
              "anneal_steps": args.anneal_steps, "seed": args.seed,
              "nested_budget_prefixes": True, "models": estimates}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.metrics, args.evidence)
    print(json.dumps({"out": str(args.out), "figure": str(args.evidence),
                      "settings": {k: result[k] for k in
                                   ("epochs", "budgets", "repeats", "anneal_steps", "test_images")}}, indent=2))


def plot(ais: dict, metrics_path: Path, output: Path):
    train = json.loads(metrics_path.read_text())
    train_models = {f"CD-{model['cd_steps']}": model for model in train["models"]}
    epochs = ais["epochs"]
    budgets = ais["budgets"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    palette = {"CD-1": "#1f77b4", "CD-10": "#d95f02"}
    budget_styles = {b: ("o", 0.55 + 0.4 * i / max(1, len(budgets) - 1))
                     for i, b in enumerate(budgets)}
    ax = axes[0]
    for model, color in palette.items():
        for budget in budgets:
            means, stds = [], []
            for epoch in epochs:
                entry = ais["models"][model][str(epoch)]["budgets"][str(budget)]
                means.append(entry["mean_nll"]); stds.append(entry["std_nll"])
            marker, alpha = budget_styles[budget]
            ax.errorbar(epochs, means, yerr=stds, color=color, marker=marker,
                        alpha=alpha, capsize=2, lw=1.4,
                        label=f"{model}, {budget} AIS particles")
    ax.set(title="Held-out NLL (AIS)", xlabel="Training epoch", ylabel="Estimated NLL (nats/image)")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)

    ax = axes[1]
    for model, model_data in train_models.items():
        h = model_data["history"]
        ax.plot([p["epoch"] for p in h[1:]], [p["cd_surrogate"] for p in h[1:]],
                color=palette[model], label=model)
    ax.set(title="Training surrogate", xlabel="Training epoch", ylabel="CD surrogate")
    ax.grid(alpha=0.25); ax.legend()

    ax = axes[2]
    for model, model_data in train_models.items():
        h = model_data["history"]
        ax.plot([p["epoch"] for p in h], [p["reconstruction_error_cd1"] for p in h],
                color=palette[model], label=model)
    ax.set(title="Held-out one-step reconstruction", xlabel="Training epoch",
           ylabel="Fraction of changed pixels")
    ax.grid(alpha=0.25); ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
