#!/usr/bin/env python3
"""Train the configured time-conditioned noise predictor with JAX on CPU."""
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

from diffusion import (
    add_noise,
    beta_schedule,
    init_noise_predictor,
    predict_noise,
    sample_target,
)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


NUM_STEPS = 1000


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    training = config["training"]
    if model["activation"].lower() != "silu":
        raise ValueError("only SiLU activation is implemented")
    if training["optimizer"].lower() != "adam":
        raise ValueError("only Adam is implemented")
    if training["learning_rate_decay"].lower() != "cosine":
        raise ValueError("only cosine learning-rate decay is implemented")
    if training["precision"].lower() != "float32":
        raise ValueError("only float32 precision is implemented")
    updates = int(training["updates"])
    batch_size = int(training["batch_size"])
    checkpoints = tuple(int(value) for value in training["checkpoint_updates"])
    if updates < 1 or batch_size < 1:
        raise ValueError("updates and batch_size must be positive")
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] != updates:
        raise ValueError("checkpoint_updates must include 0 and the final update")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoint_updates must be strictly increasing")
    if any(value < 0 or value > updates for value in checkpoints):
        raise ValueError("checkpoint update is outside the training run")
    return config


def _save_checkpoint(path: Path, params: tuple, update: int) -> None:
    arrays = {f"layer_{i}_weights": np.asarray(weights)
              for i, (weights, _) in enumerate(params)}
    arrays.update({f"layer_{i}_biases": np.asarray(biases)
                   for i, (_, biases) in enumerate(params)})
    arrays["update"] = np.asarray(update, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def _save_loss_history(path: Path, history: list[tuple[int, float, float]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("update", "noise_mse", "learning_rate"))
        writer.writerows(history)


def train(config_path: Path, output_dir: Path) -> None:
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "diffusion.toml")
    settings = {
        "config": config,
        "num_diffusion_steps": NUM_STEPS,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "target_centers": [[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
        "target_covariance": [[0.09, 0.0], [0.0, 0.09]],
        "mixture_weights": [0.25, 0.25, 0.25, 0.25],
        "objective": "mean squared error between predicted and sampled Gaussian noise",
        "noise_levels": "independent uniform integer samples from 1 through 1000 inclusive",
        "jax_backend": jax.default_backend(),
        "checkpoints_include_initialization": True,
    }
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    updates = int(train_cfg["updates"])
    batch_size = int(train_cfg["batch_size"])
    checkpoint_updates = set(int(value) for value in train_cfg["checkpoint_updates"])
    hidden_widths = tuple(int(value) for value in model_cfg["hidden_widths"])
    time_features = int(model_cfg["time_features"])
    frequency_base = float(model_cfg["time_frequency_base"])
    time_offset = float(model_cfg["time_offset"])
    beta1 = float(train_cfg["beta1"])
    beta2 = float(train_cfg["beta2"])
    epsilon = float(train_cfg["epsilon"])
    initial_lr = float(train_cfg["learning_rate"])
    final_lr = float(train_cfg["final_learning_rate"])
    betas = beta_schedule(NUM_STEPS, 0.0001, 0.02)
    alpha_bars = jnp.cumprod(1.0 - betas)

    params = init_noise_predictor(
        jax.random.PRNGKey(int(train_cfg["seed"])), hidden_widths, time_features
    )
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)

    def loss_for_key(current_params, key):
        data_key, level_key, noise_key = jax.random.split(key, 3)
        x0 = sample_target(data_key, batch_size)
        levels = jax.random.randint(level_key, (batch_size,), 1, NUM_STEPS + 1)
        noise = jax.random.normal(noise_key, x0.shape, dtype=x0.dtype)
        x_t = add_noise(x0, levels, noise, betas)
        prediction = predict_noise(
            current_params, x_t, levels, time_features, frequency_base, time_offset
        )
        return jnp.mean(jnp.square(prediction - noise))

    @jax.jit
    def update_step(current_params, m, v, step, key):
        loss, grads = jax.value_and_grad(loss_for_key)(current_params, key)
        step_f = step.astype(jnp.float32)
        m_new = jax.tree.map(lambda old, grad: beta1 * old + (1.0 - beta1) * grad, m, grads)
        v_new = jax.tree.map(lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad), v, grads)
        m_hat = jax.tree.map(lambda value: value / (1.0 - beta1 ** step_f), m_new)
        v_hat = jax.tree.map(lambda value: value / (1.0 - beta2 ** step_f), v_new)
        fraction = step_f / updates
        learning_rate = final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + jnp.cos(jnp.pi * fraction))
        new_params = jax.tree.map(
            lambda parameter, first, second: parameter - learning_rate * first / (jnp.sqrt(second) + epsilon),
            current_params, m_hat, v_hat,
        )
        return new_params, m_new, v_new, loss, learning_rate

    seed_key = jax.random.PRNGKey(int(train_cfg["seed"]))
    initial_loss = float(loss_for_key(params, jax.random.fold_in(seed_key, 0)))
    history = [(0, initial_loss, initial_lr)]
    if 0 in checkpoint_updates:
        _save_checkpoint(output_dir / "checkpoint-update-00000.npz", params, 0)
        _save_loss_history(output_dir / "loss-history.csv", history)

    started = time.perf_counter()
    for update in range(1, updates + 1):
        key = jax.random.fold_in(seed_key, update)
        params, first_moment, second_moment, loss, lr = update_step(
            params, first_moment, second_moment, jnp.asarray(update, dtype=jnp.int32), key
        )
        history.append((update, float(loss), float(lr)))
        if update in checkpoint_updates:
            checkpoint = output_dir / f"checkpoint-update-{update:05d}.npz"
            _save_checkpoint(checkpoint, params, update)
            _save_loss_history(output_dir / "loss-history.csv", history)
            print(f"update {update:>6}/{updates}: noise MSE={float(loss):.6f}; saved {checkpoint}", flush=True)

    settings["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Training complete; loss history and settings are in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("diffusion.toml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/training"))
    args = parser.parse_args()
    train(args.config, args.out)


if __name__ == "__main__":
    main()
