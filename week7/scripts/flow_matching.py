"""Flow-matching velocity models and exact Gaussian-mixture velocity."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from diffusion import CENTERS, TARGET_VARIANCE


def flow_time_embedding(times: jax.Array, num_features: int = 32,
                        frequency_base: float = 10_000.0,
                        time_scale: float = 1000.0) -> jax.Array:
    """Fourier features of continuous s using phase time_scale*s*frequency."""
    if num_features < 2 or num_features % 2:
        raise ValueError("num_features must be a positive even number")
    half = num_features // 2
    frequencies = frequency_base ** (-jnp.arange(half, dtype=jnp.float32) / half)
    phase = jnp.asarray(times, dtype=jnp.float32).reshape(-1, 1) * time_scale * frequencies
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)


def predict_velocity(params: tuple, x_s: jax.Array, times: jax.Array,
                     time_features: int = 32, frequency_base: float = 10_000.0,
                     time_scale: float = 1000.0) -> jax.Array:
    """Evaluate the time-conditioned SiLU MLP for the flow velocity."""
    x_s = jnp.asarray(x_s, dtype=jnp.float32)
    times = jnp.broadcast_to(jnp.asarray(times).reshape(-1), (x_s.shape[0],))
    embedding = flow_time_embedding(times, time_features, frequency_base, time_scale)
    hidden = jnp.concatenate((x_s, embedding), axis=-1)
    for weights, biases in params[:-1]:
        hidden = jax.nn.silu(hidden @ weights + biases)
    weights, biases = params[-1]
    return hidden @ weights + biases


def exact_mixture_velocity(x_s: jax.Array, s: float | jax.Array,
                           centers: jax.Array = CENTERS,
                           component_variance: float = TARGET_VARIANCE) -> jax.Array:
    """Compute E[z-x0 | (1-s)x0+s*z=x_s] exactly for the Gaussian mixture."""
    x_s = jnp.asarray(x_s, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=x_s.dtype)
    s = jnp.asarray(s, dtype=x_s.dtype)
    marginal_variance = jnp.square(1.0 - s) * component_variance + jnp.square(s)
    component_means = (1.0 - s) * centers
    residuals = x_s[:, None, :] - component_means[None, :, :]
    log_weights = -0.5 * jnp.sum(jnp.square(residuals), axis=-1) / marginal_variance
    responsibilities = jax.nn.softmax(log_weights, axis=-1)
    cross_covariance = s - (1.0 - s) * component_variance
    component_velocity_means = -centers[None, :, :] + (
        cross_covariance / marginal_variance
    ) * residuals
    return jnp.sum(responsibilities[:, :, None] * component_velocity_means, axis=1)


def flow_trajectory(x_noise: jax.Array, times: jax.Array,
                    velocity_predictor) -> jax.Array:
    """Euler-integrate the learned increasing-s velocity backward to s=0.

    ``times`` is a strictly descending grid starting at 1 and ending at 0.
    The output stores the initial noise state followed by every Euler state.
    """
    x_noise = jnp.asarray(x_noise, dtype=jnp.float32)
    times = jnp.asarray(times, dtype=x_noise.dtype)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must contain at least the endpoints s=1 and s=0")

    def step(x_s, pair):
        s, next_s = pair
        velocity = velocity_predictor(x_s, s)
        x_next = x_s - (s - next_s) * velocity
        return x_next, x_next

    _, later_states = lax.scan(step, x_noise, (times[:-1], times[1:]))
    return jnp.concatenate((x_noise[jnp.newaxis, ...], later_states), axis=0)
