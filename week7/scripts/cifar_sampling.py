#!/usr/bin/env python3
"""DDPM generation and RePaint inpainting for the pretrained CIFAR-10 U-Net."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def linear_betas(num_steps: int = 1000) -> np.ndarray:
    """The linear DDPM variance schedule used in the mixture exercise."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    return np.linspace(1e-4, 0.02, num_steps, dtype=np.float32)


def _alpha_bars(betas: np.ndarray) -> np.ndarray:
    return np.concatenate((np.ones((1,), dtype=np.float64), np.cumprod(1.0 - betas.astype(np.float64))))


def ddpm_sample_with_snapshots(
    key: jax.Array,
    initial: jax.Array,
    betas: np.ndarray,
    predictor,
    params,
    *,
    snapshot_levels: tuple[int, ...] = (1000, 200, 50, 0),
) -> tuple[jax.Array, dict[int, np.ndarray]]:
    """Run all adjacent DDPM steps without clipping and retain selected states.

    ``predictor(params, x_t, model_index)`` receives the checkpoint's
    zero-based model index ``t - 1``. Fresh reverse noise is drawn at every
    level except t=1, as in the mixture sampler.
    """
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim != 1 or betas.size < 1 or np.any((betas <= 0) | (betas >= 1)):
        raise ValueError("betas must be a 1-D array with entries in (0, 1)")
    num_steps = len(betas)
    if initial.ndim != 4 or initial.shape[1:] != (3, 32, 32):
        raise ValueError("initial must have shape (batch, 3, 32, 32)")
    if any(level < 0 or level > num_steps for level in snapshot_levels):
        raise ValueError("snapshot levels must be between zero and the schedule length")
    alpha_bars = _alpha_bars(betas)
    snapshots: dict[int, np.ndarray] = {}
    x_t = jnp.asarray(initial, dtype=jnp.float32)
    if num_steps in snapshot_levels:
        snapshots[num_steps] = np.asarray(x_t).copy()
    noise_key = jax.random.fold_in(key, 1)
    for t in range(num_steps, 0, -1):
        model_index = jnp.full((x_t.shape[0],), t - 1, dtype=jnp.int32)
        epsilon = predictor(params, x_t, model_index)
        beta_t = float(betas[t - 1])
        alpha_t_bar = float(alpha_bars[t])
        mean = (x_t - beta_t * epsilon / np.sqrt(1.0 - alpha_t_bar)) / np.sqrt(1.0 - beta_t)
        if t > 1:
            step_key = jax.random.fold_in(noise_key, t)
            noise = jax.random.normal(step_key, x_t.shape, dtype=jnp.float32)
            x_t = mean + np.sqrt(beta_t) * noise
        else:
            x_t = mean
        if t - 1 in snapshot_levels:
            snapshots[t - 1] = np.asarray(x_t).copy()
    jax.block_until_ready(x_t)
    return x_t, snapshots


def repaint_completion(
    key: jax.Array,
    observed: jax.Array,
    mask: jax.Array,
    betas: np.ndarray,
    predictor,
    params,
    *,
    levels: tuple[int, ...] | None = None,
    attempts_per_jump: int = 2,
) -> tuple[jax.Array, int]:
    """Inpaint one image with RePaint and return the completion/eval count.

    The mask is one for observed pixels and zero for hidden pixels. On the
    final jump to zero there is always exactly one attempt. Each Gaussian draw
    uses a distinct folded-in key; all observations are inserted at the
    matching forward-noised level after each reverse proposal.
    """
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim != 1 or betas.size < 1 or np.any((betas <= 0) | (betas >= 1)):
        raise ValueError("betas must be a 1-D array with entries in (0, 1)")
    if observed.ndim != 4 or observed.shape[1:] != (3, 32, 32):
        raise ValueError("observed must have shape (batch, 3, 32, 32)")
    if mask.shape != observed.shape:
        raise ValueError("mask must have the same shape as observed")
    if attempts_per_jump < 1:
        raise ValueError("attempts_per_jump must be positive")
    num_steps = len(betas)
    if levels is None:
        levels = tuple(range(num_steps, 0, -20)) + (0,)
    if len(levels) < 2 or levels[0] != num_steps or levels[-1] != 0:
        raise ValueError("levels must descend from the final training level to zero")
    if any(a <= b for a, b in zip(levels[:-1], levels[1:])) or any(
        level < 0 or level > num_steps for level in levels
    ):
        raise ValueError("levels must be strictly descending within the training schedule")

    alpha_bars = _alpha_bars(betas)
    observed = jnp.asarray(observed, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    draw_index = 0

    def normal(shape):
        nonlocal draw_index
        draw_key = jax.random.fold_in(key, draw_index)
        draw_index += 1
        return jax.random.normal(draw_key, shape, dtype=jnp.float32)

    x_t = normal(observed.shape)
    evaluations = 0
    for t, s in zip(levels[:-1], levels[1:]):
        ratio = float(alpha_bars[t] / alpha_bars[s])
        attempts = 1 if s == 0 else attempts_per_jump
        for attempt in range(attempts):
            if attempt > 0:
                forward_noise = normal(x_t.shape)
                x_t = np.sqrt(ratio) * x_t + np.sqrt(1.0 - ratio) * forward_noise
            model_index = jnp.full((x_t.shape[0],), t - 1, dtype=jnp.int32)
            epsilon = predictor(params, x_t, model_index)
            evaluations += 1
            x0_hat = (x_t - np.sqrt(1.0 - float(alpha_bars[t])) * epsilon) / np.sqrt(float(alpha_bars[t]))
            x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
            variance_ratio = 1.0 - ratio
            mean = (
                np.sqrt(float(alpha_bars[s])) * variance_ratio / (1.0 - float(alpha_bars[t])) * x0_hat
                + np.sqrt(ratio) * (1.0 - float(alpha_bars[s])) / (1.0 - float(alpha_bars[t])) * x_t
            )
            variance = (1.0 - float(alpha_bars[s])) * variance_ratio / (1.0 - float(alpha_bars[t]))
            proposal = mean + np.sqrt(max(variance, 0.0)) * normal(x_t.shape)
            observed_noise = normal(observed.shape)
            forward_observed = (
                np.sqrt(float(alpha_bars[s])) * observed
                + np.sqrt(1.0 - float(alpha_bars[s])) * observed_noise
            )
            x_t = mask * forward_observed + (1.0 - mask) * proposal
    jax.block_until_ready(x_t)
    return x_t, evaluations
