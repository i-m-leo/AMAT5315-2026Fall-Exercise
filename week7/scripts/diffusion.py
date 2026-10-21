#!/usr/bin/env python3
"""JAX sampling for a four-component Gaussian target and its forward process."""
from __future__ import annotations

import os

# The exercise is CPU-first; callers may explicitly choose another backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import lax

CENTERS = jnp.array([[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
                    dtype=jnp.float32)
TARGET_VARIANCE = 0.09


def sample_target(key: jax.Array, num_samples: int) -> jax.Array:
    """Draw equally weighted samples from four N(center, 0.09 I) components."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    component_key, noise_key = jax.random.split(key)
    components = jax.random.randint(component_key, (num_samples,), 0, len(CENTERS))
    noise = jax.random.normal(noise_key, (num_samples, 2), dtype=jnp.float32)
    return CENTERS[components] + jnp.sqrt(TARGET_VARIANCE) * noise


def beta_schedule(num_steps: int = 1000, beta_start: float = 0.0001,
                  beta_end: float = 0.02) -> jax.Array:
    """Return the linear per-step noise-variance schedule beta_1,...,beta_T."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("require 0 < beta_start <= beta_end < 1")
    return jnp.linspace(beta_start, beta_end, num_steps, dtype=jnp.float32)


def q_sample(key: jax.Array, x0: jax.Array, t: int | jax.Array,
             betas: jax.Array) -> jax.Array:
    """Sample the closed-form marginal q(x_t | x_0) at 1-based level t.

    ``t`` may be a scalar shared by the batch or a vector with one level per
    row of ``x0``. The forward marginal is N(sqrt(alpha_bar_t) x0,
    (1-alpha_bar_t) I), where alpha_bar_t = product_{s=1}^t (1-beta_s).
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    betas = jnp.asarray(betas, dtype=x0.dtype)
    levels = jnp.asarray(t, dtype=jnp.int32)
    alpha_bar = jnp.cumprod(1.0 - betas)[levels - 1]
    while alpha_bar.ndim < x0.ndim:
        alpha_bar = alpha_bar[..., None]
    noise = jax.random.normal(key, x0.shape, dtype=x0.dtype)
    return jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise


def forward_diffusion(key: jax.Array, x0: jax.Array,
                      betas: jax.Array) -> jax.Array:
    """Generate x_0,...,x_T by applying each forward Gaussian transition.

    The returned array has shape ``(T + 1, *x0.shape)`` and starts with the
    input unchanged. Each transition is x_t = sqrt(1-beta_t) x_(t-1)
    + sqrt(beta_t) epsilon_t for independent standard-normal epsilon_t.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    betas = jnp.asarray(betas, dtype=x0.dtype)
    step_keys = jax.random.split(key, betas.shape[0])

    def transition(previous, inputs):
        beta, step_key = inputs
        noise = jax.random.normal(step_key, previous.shape, dtype=previous.dtype)
        current = jnp.sqrt(1.0 - beta) * previous + jnp.sqrt(beta) * noise
        return current, current

    _, later_states = lax.scan(transition, x0, (betas, step_keys))
    return jnp.concatenate((x0[jnp.newaxis, ...], later_states), axis=0)
