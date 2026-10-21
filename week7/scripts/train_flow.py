#!/usr/bin/env python3
"""Train the flow-matching velocity predictor configured in flow.toml."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from diffusion import CENTERS, TARGET_VARIANCE, init_noise_predictor, sample_target
from flow_matching import predict_velocity

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def _save_checkpoint(path: Path, params: tuple, update: int) -> None:
    arrays = {f"layer_{i}_weights": np.asarray(weights)
              for i, (weights, _) in enumerate(params)}
    arrays.update({f"layer_{i}_biases": np.asarray(biases)
                   for i, (_, biases) in enumerate(params)})
    arrays["update"] = np.asarray(update, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def _save_history(path: Path, history: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("update", "velocity_mse", "learning_rate"))
        writer.writerows(history)


def train(config_path: Path, output_dir: Path) -> None:
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    training = config["training"]
    if model["activation"].lower() != "silu" or training["optimizer"].lower() != "adam":
        raise ValueError("flow implementation supports the configured SiLU MLP with Adam")
    if training["learning_rate_decay"].lower() != "cosine":
        raise ValueError("flow implementation requires cosine learning-rate decay")
    if training["precision"].lower() != "float32":
        raise ValueError("flow implementation supports float32 precision")

    updates = int(training["updates"])
    batch_size = int(training["batch_size"])
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "flow.toml")
    settings = {
        "config": config,
        "target_centers": np.asarray(CENTERS).tolist(),
        "target_covariance": [[TARGET_VARIANCE, 0.0], [0.0, TARGET_VARIANCE]],
        "target_mixture_weights": [0.25] * 4,
        "path": "x_s=(1-s)*x0+s*z; x0~target, z~N(0,I), s~Uniform[0,1]",
        "target_velocity": "z-x0",
        "loss": "mean squared error averaged over both velocity coordinates",
        "jax_backend": jax.default_backend(),
        "saved_updates": [0, updates],
    }
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    widths = tuple(int(width) for width in model["hidden_widths"])
    n_features = int(model["time_features"])
    frequency_base = float(model["time_frequency_base"])
    time_scale = float(model["time_scale"])
    beta1 = float(training["beta1"])
    beta2 = float(training["beta2"])
    epsilon = float(training["epsilon"])
    initial_lr = float(training["learning_rate"])
    final_lr = float(training["final_learning_rate"])
    seed = int(training["seed"])
    params = init_noise_predictor(jax.random.PRNGKey(seed), widths, n_features)
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)

    def loss_for_key(current_params, key):
        data_key, noise_key, time_key = jax.random.split(key, 3)
        x0 = sample_target(data_key, batch_size)
        z = jax.random.normal(noise_key, x0.shape, dtype=x0.dtype)
        s = jax.random.uniform(time_key, (batch_size,), minval=0.0, maxval=1.0)
        x_s = (1.0 - s[:, None]) * x0 + s[:, None] * z
        target_velocity = z - x0
        prediction = predict_velocity(current_params, x_s, s, n_features,
                                      frequency_base, time_scale)
        return jnp.mean(jnp.square(prediction - target_velocity))

    @jax.jit
    def update_step(current_params, m, v, step, key):
        loss, grads = jax.value_and_grad(loss_for_key)(current_params, key)
        step_f = step.astype(jnp.float32)
        m_new = jax.tree.map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads)
        v_new = jax.tree.map(lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad), v, grads)
        m_hat = jax.tree.map(lambda value: value / (1.0 - beta1 ** step_f), m_new)
        v_hat = jax.tree.map(lambda value: value / (1.0 - beta2 ** step_f), v_new)
        fraction = step_f / updates
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + jnp.cos(jnp.pi * fraction))
        new_params = jax.tree.map(
            lambda parameter, first, second: parameter - lr * first / (jnp.sqrt(second) + epsilon),
            current_params, m_hat, v_hat,
        )
        return new_params, m_new, v_new, loss, lr

    base_key = jax.random.PRNGKey(seed)
    history = [(0, float(loss_for_key(params, jax.random.fold_in(base_key, 0))), initial_lr)]
    _save_checkpoint(output_dir / "checkpoint-update-00000.npz", params, 0)
    _save_history(output_dir / "loss-history.csv", history)
    started = time.perf_counter()
    for update in range(1, updates + 1):
        key = jax.random.fold_in(base_key, update)
        params, first_moment, second_moment, loss, lr = update_step(
            params, first_moment, second_moment, jnp.asarray(update, dtype=jnp.int32), key
        )
        history.append((update, float(loss), float(lr)))
        if update % 1000 == 0 or update == updates:
            print(f"update {update:>6}/{updates}: velocity MSE={float(loss):.6f}", flush=True)
    _save_checkpoint(output_dir / f"checkpoint-update-{updates:05d}.npz", params, updates)
    _save_history(output_dir / "loss-history.csv", history)
    settings["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved flow model, settings, and loss history to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("flow.toml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/flow-training"))
    args = parser.parse_args()
    train(args.config, args.out)


if __name__ == "__main__":
    main()
