#!/usr/bin/env python3
"""JAX Gaussian-mixture diffusion, time-conditioned prediction, and DDPM sampling."""
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


def add_noise(x0: jax.Array, t: int | jax.Array, noise: jax.Array,
              betas: jax.Array) -> jax.Array:
    """Apply the closed-form forward marginal at 1-based level(s) ``t``."""
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    noise = jnp.asarray(noise, dtype=x0.dtype)
    betas = jnp.asarray(betas, dtype=x0.dtype)
    levels = jnp.asarray(t, dtype=jnp.int32)
    alpha_bar = jnp.cumprod(1.0 - betas)[levels - 1]
    while alpha_bar.ndim < x0.ndim:
        alpha_bar = alpha_bar[..., None]
    return jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise


def q_sample(key: jax.Array, x0: jax.Array, t: int | jax.Array,
             betas: jax.Array) -> jax.Array:
    """Sample q(x_t | x_0), with scalar or per-example 1-based levels.

    The marginal is N(sqrt(alpha_bar_t) x0, (1-alpha_bar_t) I), where
    alpha_bar_t = product_{s=1}^t (1-beta_s).
    """
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    noise = jax.random.normal(key, x0.shape, dtype=x0.dtype)
    return add_noise(x0, t, noise, betas)


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


def time_embedding(levels: jax.Array, num_features: int = 32,
                   frequency_base: float = 10_000.0,
                   time_offset: float = -1.0) -> jax.Array:
    """Sin/cos Fourier embedding of 1-based diffusion levels."""
    if num_features < 2 or num_features % 2:
        raise ValueError("num_features must be a positive even number")
    half = num_features // 2
    frequencies = frequency_base ** (
        -jnp.arange(half, dtype=jnp.float32) / half
    )
    phase = (jnp.asarray(levels, dtype=jnp.float32).reshape(-1, 1) + time_offset) * frequencies
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def init_noise_predictor(key: jax.Array, hidden_widths: tuple[int, ...] | list[int],
                         time_features: int = 32) -> tuple:
    """Initialize an MLP with N(0, 1/sqrt(fan_in)) weights and zero biases."""
    if not hidden_widths or any(width < 1 for width in hidden_widths):
        raise ValueError("hidden_widths must contain positive widths")
    dimensions = (2 + time_features, *hidden_widths, 2)
    keys = jax.random.split(key, len(dimensions) - 1)
    layers = []
    for layer_key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        weights = jax.random.normal(layer_key, (fan_in, fan_out), dtype=jnp.float32)
        weights = weights / jnp.sqrt(jnp.asarray(fan_in, dtype=jnp.float32))
        biases = jnp.zeros((fan_out,), dtype=jnp.float32)
        layers.append((weights, biases))
    return tuple(layers)


def predict_noise(params: tuple, x_t: jax.Array, levels: int | jax.Array,
                  time_features: int = 32, frequency_base: float = 10_000.0,
                  time_offset: float = -1.0) -> jax.Array:
    """Predict epsilon from noisy points and their diffusion levels using SiLU."""
    x_t = jnp.asarray(x_t, dtype=jnp.float32)
    levels = jnp.broadcast_to(jnp.asarray(levels).reshape(-1), (x_t.shape[0],))
    embedding = time_embedding(levels, time_features, frequency_base, time_offset)
    hidden = jnp.concatenate((x_t, embedding), axis=-1)
    for weights, biases in params[:-1]:
        hidden = jax.nn.silu(hidden @ weights + biases)
    weights, biases = params[-1]
    return hidden @ weights + biases


def ddpm_sample(key: jax.Array, params: tuple, num_samples: int,
                betas: jax.Array, time_features: int = 32,
                frequency_base: float = 10_000.0,
                time_offset: float = -1.0) -> jax.Array:
    """Generate x_0 using DDPM reverse steps with variance beta_t.

    No predicted x_0 clipping is applied. The stochastic term is omitted at
    t=1, so the final reverse step is deterministic given x_1 and the model.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    betas = jnp.asarray(betas, dtype=jnp.float32)
    num_steps = betas.shape[0]
    alpha = 1.0 - betas
    alpha_bar = jnp.cumprod(alpha)
    initial_key, steps_key = jax.random.split(key)
    x_terminal = jax.random.normal(initial_key, (num_samples, 2), dtype=jnp.float32)
    step_keys = jax.random.split(steps_key, num_steps)
    reverse_indices = jnp.arange(num_steps - 1, -1, -1)

    def reverse_step(x_t, inputs):
        index, step_key = inputs
        level = index + 1
        epsilon = predict_noise(params, x_t, level, time_features,
                                frequency_base, time_offset)
        beta_t = betas[index]
        alpha_t = alpha[index]
        alpha_bar_t = alpha_bar[index]
        mean = (x_t - beta_t * epsilon / jnp.sqrt(1.0 - alpha_bar_t)) / jnp.sqrt(alpha_t)
        noise = jax.random.normal(step_key, x_t.shape, dtype=x_t.dtype)
        previous = mean + jnp.where(index > 0, jnp.sqrt(beta_t) * noise, 0.0)
        return previous, None

    final, _ = lax.scan(reverse_step, x_terminal, (reverse_indices, step_keys))
    return final
