from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from diffusion import (
    CENTERS,
    TARGET_VARIANCE,
    beta_schedule,
    ddpm_sample,
    forward_diffusion,
    init_noise_predictor,
    predict_noise,
    q_sample,
    sample_target,
)


def test_target_samples_follow_equally_weighted_gaussian_mixture():
    samples = np.asarray(sample_target(jax.random.PRNGKey(17), 20_000))
    assert samples.shape == (20_000, 2)
    nearest = np.linalg.norm(samples[:, None, :] - np.asarray(CENTERS)[None, :, :], axis=-1).argmin(axis=1)
    counts = np.bincount(nearest, minlength=4)
    np.testing.assert_allclose(counts / len(samples), np.full(4, 0.25), atol=0.015)

    for component, center in enumerate(np.asarray(CENTERS)):
        residual = samples[nearest == component] - center
        np.testing.assert_allclose(residual.mean(axis=0), 0.0, atol=0.02)
        np.testing.assert_allclose(np.cov(residual, rowvar=False), TARGET_VARIANCE * np.eye(2), atol=0.006)


def test_target_sampling_is_reproducible_and_validates_sample_count():
    key = jax.random.PRNGKey(4)
    np.testing.assert_array_equal(sample_target(key, 10), sample_target(key, 10))
    with pytest.raises(ValueError, match="positive"):
        sample_target(key, 0)


def test_beta_schedule_is_linear_with_requested_endpoints():
    betas = beta_schedule()
    assert betas.shape == (1000,)
    np.testing.assert_allclose(np.asarray(betas)[[0, -1]], [0.0001, 0.02], rtol=1e-6)
    assert np.all(np.diff(np.asarray(betas)) > 0)
    np.testing.assert_allclose(np.asarray(beta_schedule(5, 0.01, 0.05)),
                               np.linspace(0.01, 0.05, 5), rtol=1e-6)
    with pytest.raises(ValueError):
        beta_schedule(0)


def test_q_sample_matches_closed_form_for_per_example_levels():
    x0 = jnp.array([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    betas = beta_schedule(5, 0.01, 0.05)
    levels = jnp.array([1, 4])
    key = jax.random.PRNGKey(9)
    actual = q_sample(key, x0, levels, betas)
    alpha_bar = jnp.cumprod(1.0 - betas)[levels - 1, None]
    noise = jax.random.normal(key, x0.shape, dtype=x0.dtype)
    expected = jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_forward_trajectory_starts_at_input_and_reaches_marginal_variance():
    betas = beta_schedule()
    x0 = jnp.zeros((1024, 2), dtype=jnp.float32)
    trajectory = np.asarray(forward_diffusion(jax.random.PRNGKey(21), x0, betas))
    assert trajectory.shape == (1001, 1024, 2)
    np.testing.assert_array_equal(trajectory[0], np.zeros((1024, 2), dtype=np.float32))

    expected_variance = 1.0 - float(jnp.prod(1.0 - betas))
    np.testing.assert_allclose(trajectory[-1].mean(axis=0), 0.0, atol=0.08)
    np.testing.assert_allclose(trajectory[-1].var(axis=0), expected_variance, atol=0.10)


def test_time_conditioned_noise_predictor_shapes_and_depends_on_level():
    params = init_noise_predictor(jax.random.PRNGKey(2), [8, 8], time_features=4)
    x_t = jnp.ones((3, 2), dtype=jnp.float32)
    predictions = predict_noise(params, x_t, jnp.array([1, 20, 100]), time_features=4)
    assert predictions.shape == (3, 2)
    assert np.isfinite(np.asarray(predictions)).all()
    assert not np.allclose(np.asarray(predictions[0]), np.asarray(predictions[1]))


def test_ddpm_uses_beta_variance_without_clipping_or_final_noise():
    betas = beta_schedule(3, 0.01, 0.03)
    params = init_noise_predictor(jax.random.PRNGKey(5), [8], time_features=4)
    params = jax.tree.map(jnp.zeros_like, params)  # Predict epsilon == 0 at every level.
    key = jax.random.PRNGKey(13)
    actual = ddpm_sample(key, params, 64, betas, time_features=4)

    initial_key, steps_key = jax.random.split(key)
    current = jax.random.normal(initial_key, (64, 2), dtype=jnp.float32)
    step_keys = jax.random.split(steps_key, len(betas))
    for scan_index, level_index in enumerate((2, 1, 0)):
        beta = betas[level_index]
        current = current / jnp.sqrt(1.0 - beta)
        if level_index > 0:
            noise = jax.random.normal(step_keys[scan_index], current.shape, dtype=current.dtype)
            current = current + jnp.sqrt(beta) * noise
    np.testing.assert_allclose(np.asarray(actual), np.asarray(current), rtol=1e-6, atol=1e-6)
    assert np.max(np.abs(np.asarray(actual))) > 1.0  # No x_0 clipping.
