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


def ddpm_trajectory(params: tuple, x_terminal: jax.Array,
                    reverse_noises: jax.Array, betas: jax.Array,
                    time_features: int = 32, frequency_base: float = 10_000.0,
                    time_offset: float = -1.0) -> jax.Array:
    """Return x_T,...,x_0 using caller-supplied noise for matched trajectories.

    ``reverse_noises`` is ordered by reverse scan step (t=T,...,1), with shape
    ``(T, *x_terminal.shape)``. Its last entry, corresponding to t=1, is
    intentionally ignored. No x_0 clipping is applied.
    """
    x_terminal = jnp.asarray(x_terminal, dtype=jnp.float32)
    betas = jnp.asarray(betas, dtype=jnp.float32)
    reverse_noises = jnp.asarray(reverse_noises, dtype=x_terminal.dtype)
    num_steps = betas.shape[0]
    if reverse_noises.shape != (num_steps, *x_terminal.shape):
        raise ValueError("reverse_noises must have shape (T, *x_terminal.shape)")
    alpha = 1.0 - betas
    alpha_bar = jnp.cumprod(alpha)
    reverse_indices = jnp.arange(num_steps - 1, -1, -1)

    def reverse_step(x_t, inputs):
        index, noise = inputs
        level = index + 1
        epsilon = predict_noise(params, x_t, level, time_features,
                                frequency_base, time_offset)
        beta_t = betas[index]
        alpha_t = alpha[index]
        alpha_bar_t = alpha_bar[index]
        mean = (x_t - beta_t * epsilon / jnp.sqrt(1.0 - alpha_bar_t)) / jnp.sqrt(alpha_t)
        previous = mean + jnp.where(index > 0, jnp.sqrt(beta_t) * noise, 0.0)
        return previous, previous

    _, later_states = lax.scan(reverse_step, x_terminal, (reverse_indices, reverse_noises))
    return jnp.concatenate((x_terminal[jnp.newaxis, ...], later_states), axis=0)


def ddpm_sample(key: jax.Array, params: tuple, num_samples: int,
                betas: jax.Array, time_features: int = 32,
                frequency_base: float = 10_000.0,
                time_offset: float = -1.0) -> jax.Array:
    """Generate x_0 using DDPM reverse steps with variance beta_t."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    betas = jnp.asarray(betas, dtype=jnp.float32)
    initial_key, steps_key = jax.random.split(key)
    x_terminal = jax.random.normal(initial_key, (num_samples, 2), dtype=jnp.float32)
    step_keys = jax.random.split(steps_key, betas.shape[0])
    reverse_noises = jax.vmap(
        lambda step_key: jax.random.normal(step_key, x_terminal.shape, dtype=x_terminal.dtype)
    )(step_keys)
    trajectory = ddpm_trajectory(params, x_terminal, reverse_noises, betas,
                                 time_features, frequency_base, time_offset)
    return trajectory[-1]


def select_ddim_timesteps(num_inference_steps: int,
                          total_steps: int = 1000) -> jax.Array:
    """Choose evenly spaced descending levels, including T and level 1."""
    if total_steps < 1 or not 1 <= num_inference_steps <= total_steps:
        raise ValueError("num_inference_steps must be in [1, total_steps]")
    return jnp.rint(jnp.linspace(total_steps, 1, num_inference_steps)).astype(jnp.int32)


def ddim_trajectory(x_terminal: jax.Array, timesteps: jax.Array,
                    betas: jax.Array, noise_predictor) -> jax.Array:
    """Deterministic eta=0 DDIM trajectory for selected descending levels.

    ``timesteps`` contains selected 1-based levels in descending order,
    starting at T. At each selected level t, this jumps to the next selected
    level s (or s=0 at the end) using
    x_s = sqrt(alpha_bar_s) * x0_hat + sqrt(1-alpha_bar_s) * epsilon_hat.
    The returned states are ordered x_T,...,x_0. No random noise is added.
    ``noise_predictor(x_t, t)`` must return a batch of epsilon predictions.
    """
    x_terminal = jnp.asarray(x_terminal, dtype=jnp.float32)
    betas = jnp.asarray(betas, dtype=x_terminal.dtype)
    timesteps = jnp.asarray(timesteps, dtype=jnp.int32)
    if x_terminal.ndim != 2 or x_terminal.shape[-1] != 2:
        raise ValueError("x_terminal must have shape (batch, 2)")
    if timesteps.ndim != 1 or timesteps.shape[0] < 1:
        raise ValueError("timesteps must be a non-empty 1-D array")
    alpha_bar = jnp.concatenate((jnp.ones((1,), dtype=betas.dtype),
                                 jnp.cumprod(1.0 - betas)))
    previous_times = jnp.concatenate((timesteps[1:], jnp.zeros((1,), dtype=jnp.int32)))

    def step(x_t, times):
        t, s = times
        epsilon = noise_predictor(x_t, t)
        alpha_t = alpha_bar[t]
        alpha_s = alpha_bar[s]
        x0_pred = (x_t - jnp.sqrt(1.0 - alpha_t) * epsilon) / jnp.sqrt(alpha_t)
        x_s = jnp.sqrt(alpha_s) * x0_pred + jnp.sqrt(1.0 - alpha_s) * epsilon
        return x_s, x_s

    _, later_states = lax.scan(step, x_terminal, (timesteps, previous_times))
    return jnp.concatenate((x_terminal[jnp.newaxis, ...], later_states), axis=0)


def ddim_sample(x_terminal: jax.Array, timesteps: jax.Array,
                betas: jax.Array, noise_predictor) -> jax.Array:
    """Return the final state of deterministic selected-step DDIM sampling."""
    return ddim_trajectory(x_terminal, timesteps, betas, noise_predictor)[-1]


def mixture_conditional_mean_noise(x_t: jax.Array, level: int | jax.Array,
                                   betas: jax.Array,
                                   centers: jax.Array = CENTERS,
                                   component_variance: float = TARGET_VARIANCE) -> jax.Array:
    """Exact E[epsilon | x_t] for the equally weighted isotropic Gaussian mixture.

    Each component's forward marginal is Gaussian with mean
    sqrt(alpha_bar_t)*center and variance
    (1-alpha_bar_t) + alpha_bar_t*component_variance. Posterior component
    responsibilities weight the componentwise Gaussian conditional means.
    ``x_t`` has shape (batch, 2); ``level`` is a shared 1-based level.
    """
    x_t = jnp.asarray(x_t, dtype=jnp.float32)
    betas = jnp.asarray(betas, dtype=x_t.dtype)
    centers = jnp.asarray(centers, dtype=x_t.dtype)
    alpha_bar = jnp.cumprod(1.0 - betas)[jnp.asarray(level, dtype=jnp.int32) - 1]
    sqrt_alpha_bar = jnp.sqrt(alpha_bar)
    marginal_variance = (1.0 - alpha_bar) + alpha_bar * component_variance
    component_xt_means = sqrt_alpha_bar * centers
    residuals = x_t[:, None, :] - component_xt_means[None, :, :]
    log_responsibilities = -0.5 * jnp.sum(jnp.square(residuals), axis=-1) / marginal_variance
    responsibilities = jax.nn.softmax(log_responsibilities, axis=-1)
    conditional_x0_by_component = centers[None, :, :] + (
        sqrt_alpha_bar * component_variance / marginal_variance
    ) * residuals
    conditional_x0 = jnp.sum(
        responsibilities[:, :, None] * conditional_x0_by_component, axis=1
    )
    return (x_t - sqrt_alpha_bar * conditional_x0) / jnp.sqrt(1.0 - alpha_bar)
