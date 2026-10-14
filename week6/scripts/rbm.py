#!/usr/bin/env python3
"""Train CPU JAX binary RBMs on MNIST with CD-1 and CD-10."""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def load_mnist(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = "train" if split == "train" else "t10k"
    with gzip.open(directory / f"{prefix}-images-idx3-ubyte.gz", "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("invalid MNIST image IDX magic")
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(directory / f"{prefix}-labels-idx1-ubyte.gz", "rb") as f:
        label_magic, label_count = struct.unpack(">II", f.read(8))
        if label_magic != 2049:
            raise ValueError("invalid MNIST label IDX magic")
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
    if raw.size != count * rows * cols or label_count != count or len(labels) != count:
        raise ValueError(f"inconsistent MNIST {split} images and labels")
    # Half-intensity threshold: values 128..255 map to 1; 0..127 map to 0.
    return (raw.reshape(count, rows * cols) >= 128).astype(np.float32), labels


def _log1pexp(x):
    return jnp.logaddexp(jnp.array(0, dtype=x.dtype), x)


def energy(v, h, params):
    """E(v,h) = -vb·v - hb·h - vᵀWh (batch dimensions allowed)."""
    w, vb, hb = params
    return -(jnp.sum(v * vb, axis=-1) + jnp.sum(h * hb, axis=-1)
             + jnp.sum((v @ w) * h, axis=-1))


def visible_free_energy(v, params):
    """F(v) = -vb·v - sum_j softplus(hb_j + vW_j)."""
    w, vb, hb = params
    return -(jnp.sum(v * vb, axis=-1) + jnp.sum(_log1pexp(v @ w + hb), axis=-1))


def gibbs_vhv(key, v, params):
    """One binary visible→hidden→visible block Gibbs transition."""
    w, vb, hb = params
    kh, kv = jax.random.split(key)
    ph = jax.nn.sigmoid(v @ w + hb)
    h = jax.random.bernoulli(kh, ph).astype(v.dtype)
    pv = jax.nn.sigmoid(h @ w.T + vb)
    v_next = jax.random.bernoulli(kv, pv).astype(v.dtype)
    return v_next, h, pv


def gibbs_steps(key, v, params, steps: int):
    if steps < 1:
        raise ValueError("Gibbs steps must be positive")

    def body(i, state):
        k, current = state
        k, draw_key = jax.random.split(k)
        nxt, _, _ = gibbs_vhv(draw_key, current, params)
        return k, nxt

    _, final = lax.fori_loop(0, steps, body, (key, v))
    return final


def cd_surrogate(params, v_data, key, steps: int):
    """Mean batch CD surrogate; stop gradient through sampled negative states."""
    v_model = lax.stop_gradient(gibbs_steps(key, v_data, params, steps))
    return jnp.mean(visible_free_energy(v_data, params)
                    - visible_free_energy(v_model, params))


def init_params(key, n_visible: int, n_hidden: int, train_data: np.ndarray,
                weight_std: float, hidden_bias: float, clip: tuple[float, float]):
    kw, = jax.random.split(key, 1)
    w = weight_std * jax.random.normal(kw, (n_visible, n_hidden), dtype=jnp.float32)
    p = np.clip(train_data.mean(axis=0), clip[0], clip[1])
    vb = jnp.asarray(np.log(p / (1 - p)), dtype=jnp.float32)
    hb = jnp.full((n_hidden,), hidden_bias, dtype=jnp.float32)
    return w, vb, hb


def _metrics(params, test_data, keys):
    free = visible_free_energy(jnp.asarray(test_data), params)
    result = {"free_energy_mean": float(jnp.mean(free))}
    # Fixed initial test configurations and fixed random streams per eval call.
    for steps, key in keys.items():
        changed = []
        for i, row in enumerate(test_data):
            k = jax.random.fold_in(key, i)
            recon = gibbs_steps(k, jnp.asarray(row[None, :]), params, steps)
            changed.append(jnp.mean(recon[0] != row))
        result[f"reconstruction_error_cd{steps}"] = float(jnp.mean(jnp.stack(changed)))
    return result


def train_one(train_data: np.ndarray, test_data: np.ndarray, cfg: dict,
              steps: int) -> dict:
    tr = cfg["training"]
    ini = cfg["initialization"]
    ev = cfg["evaluation"]
    params = init_params(jax.random.PRNGKey(ini["seed"]), train_data.shape[1], 128,
                         train_data[:int(tr["images"])], float(ini["weight_std"]),
                         float(ini["hidden_bias"]), tuple(map(float, ini["pixel_frequency_clip"])))
    params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)
    optimizer = str(tr["optimizer"]).upper()
    if optimizer != "SGD" or float(tr["momentum"]) != 0 or float(tr["weight_decay"]) != 0:
        raise ValueError("this implementation supports plain SGD without momentum or weight decay")

    n_train = min(int(tr["images"]), len(train_data))
    train = train_data[:n_train]
    test = test_data[:int(ev["test_images"])]
    if len(test) == 0:
        raise ValueError("evaluation set is empty")
    base_key = jax.random.PRNGKey(int(tr["seed"]))
    eval_seed = int(ev["seed"])
    def eval_keys(_epoch):
        # Reuse identical keys at initialization and every epoch, and across CD-1/CD-10.
        return {k: jax.random.fold_in(jax.random.PRNGKey(eval_seed), k) for k in (1, 10)}

    @jax.jit
    def update(p, batch, key):
        loss, grads = jax.value_and_grad(cd_surrogate)(p, batch, key, steps)
        new_p = jax.tree.map(lambda x, g: x - float(tr["learning_rate"]) * g, p, grads)
        return new_p, loss

    history = [{"epoch": 0, **_metrics(params, test, eval_keys(0))}]
    checkpoints = {0: jax.tree.map(lambda x: np.asarray(x).copy(), params)}
    batch_size = int(tr["batch_size"])
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for epoch in range(int(tr["epochs"])):
        order_key = jax.random.fold_in(base_key, epoch)
        order = np.asarray(jax.random.permutation(order_key, n_train)) if tr["shuffle_each_epoch"] else np.arange(n_train)
        epoch_keys = jax.random.fold_in(base_key, 1_000_000 + epoch)
        losses = []
        for batch_index, start in enumerate(range(0, n_train, batch_size)):
            batch = jnp.asarray(train[order[start:start + batch_size]])
            key = jax.random.fold_in(epoch_keys, batch_index)
            params, loss = update(params, batch, key)
            losses.append(float(loss))
        history.append({"epoch": epoch + 1,
                        "cd_surrogate": float(np.mean(losses)),
                        **_metrics(params, test, eval_keys(epoch + 1))})
        if epoch + 1 in (10, 20):
            checkpoints[epoch + 1] = jax.tree.map(lambda x: np.asarray(x).copy(), params)
    return {"cd_steps": steps, "history": history, "parameters": params,
            "checkpoints": checkpoints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("rbm.toml"))
    parser.add_argument("--data", type=Path, default=Path("inputs/mnist"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/rbm"))
    parser.add_argument("--models", type=int, nargs="+", choices=(1, 10), default=(1, 10),
                        help="CD-k models to train; use --models 10 to train only missing CD-10")
    args = parser.parse_args()
    with args.config.open("rb") as f:
        cfg = tomllib.load(f)
    train_x, train_y = load_mnist(args.data, "train")
    test_x, test_y = load_mnist(args.data, "test")
    args.out.mkdir(parents=True, exist_ok=True)
    # Save a complete settings snapshot alongside machine-readable results.
    (args.out / "rbm.toml").write_text(args.config.read_text())
    results = {"settings": cfg, "dataset": "MNIST", "binarization": "uint8 >= 128",
               "surrogate": "mean_batch(mean(F(v_data) - F(stop_gradient(v_k)))) with k binary block-Gibbs transitions; differentiate both free-energy terms with respect to parameters, not sampled configurations",
               "hidden_units": 128, "train_examples_used": min(int(cfg["training"]["images"]), len(train_x)),
               "test_examples_used": min(int(cfg["evaluation"]["test_images"]), len(test_x)),
               "models": []}
    metrics_path = args.out / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text())
        results["models"] = [m for m in previous.get("models", [])
                              if int(m["cd_steps"]) not in args.models]
    for steps in args.models:
        model = train_one(train_x, test_x, cfg, steps)
        model["test_labels"] = test_y[:int(cfg["evaluation"]["test_images"])].tolist()
        # Store learned parameters for downstream visualization/reproducibility.
        final_params = model.pop("parameters")
        checkpoint_params = model.pop("checkpoints")
        names = ("weights", "visible_bias", "hidden_bias")
        np.savez_compressed(args.out / f"cd{steps}-parameters.npz",
                            **dict(zip(names, map(np.asarray, final_params))))
        for epoch, values in checkpoint_params.items():
            np.savez_compressed(args.out / f"cd{steps}-epoch{epoch:02d}.npz",
                                **dict(zip(names, values)))
        results["models"].append(model)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "models": [
        {"cd_steps": m["cd_steps"], "initial": m["history"][0], "final": m["history"][-1]}
        for m in results["models"]]}, indent=2))


if __name__ == "__main__":
    main()
