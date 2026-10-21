from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from mode_metrics import assign_nearest_mode, summarize_modes
from flow_matching import exact_mixture_velocity, flow_time_embedding, flow_trajectory, predict_velocity
from diffusion import (
    CENTERS,
    TARGET_VARIANCE,
    beta_schedule,
    ddpm_sample,
    ddpm_trajectory,
    ddpm_subsample_trajectory,
    ddim_sample,
    ddim_trajectory,
    forward_diffusion,
    init_noise_predictor,
    mixture_conditional_mean_noise,
    predict_noise,
    q_sample,
    sample_target,
    select_ddim_timesteps,
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


def test_subsampled_ddpm_matches_adjacent_ddpm_on_full_grid():
    betas = beta_schedule(4, 0.01, 0.04)
    params = init_noise_predictor(jax.random.PRNGKey(41), [4], time_features=4)
    params = jax.tree.map(jnp.zeros_like, params)
    initial = jnp.array([[0.4, -0.7]], dtype=jnp.float32)
    noises = jnp.arange(8, dtype=jnp.float32).reshape(4, 1, 2) / 10
    full = ddpm_trajectory(params, initial, noises, betas, time_features=4)
    grid = ddpm_subsample_trajectory(params, initial, jnp.array([4, 3, 2, 1, 0]),
                                     noises, betas, time_features=4)
    np.testing.assert_allclose(np.asarray(full), np.asarray(grid), rtol=1e-6, atol=1e-6)


def test_flow_embedding_velocity_and_exact_marginal_endpoints():
    times = jnp.array([0.1, 0.5, 0.9])
    assert flow_time_embedding(times).shape == (3, 32)
    params = init_noise_predictor(jax.random.PRNGKey(45), [8], time_features=4)
    prediction = predict_velocity(params, jnp.ones((3, 2)), times, time_features=4)
    assert prediction.shape == (3, 2)
    x = jnp.array([[-2.0, 1.0], [0.25, -0.5]], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(exact_mixture_velocity(x, 0.0)), -np.asarray(x), atol=1e-6)
    np.testing.assert_allclose(np.asarray(exact_mixture_velocity(x, 1.0)), np.asarray(x), atol=1e-6)


def test_flow_euler_trajectory_uses_decreasing_time_grid():
    x1 = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    times = jnp.array([1.0, 0.5, 0.0])
    velocity = lambda x, s: jnp.ones_like(x) * s
    path = flow_trajectory(x1, times, velocity)
    expected = jnp.array([[[1.0, -2.0]], [[0.5, -2.5]], [[0.25, -2.75]]])
    np.testing.assert_allclose(np.asarray(path), np.asarray(expected), atol=1e-6)


def test_ddim_levels_are_selectable_and_descending():
    levels = np.asarray(select_ddim_timesteps(5, 1000))
    assert levels[0] == 1000 and levels[-1] == 1
    assert np.all(np.diff(levels) < 0)
    assert len(select_ddim_timesteps(1000, 1000)) == 1000
    with pytest.raises(ValueError):
        select_ddim_timesteps(0, 1000)
    with pytest.raises(ValueError):
        select_ddim_timesteps(1001, 1000)


def test_ddim_trajectory_is_deterministic_for_selected_levels():
    betas = beta_schedule(4, 0.01, 0.04)
    levels = jnp.array([4, 2], dtype=jnp.int32)
    initial = jnp.array([[0.4, -0.7], [-0.2, 0.3]], dtype=jnp.float32)
    def predictor(x_t, _level):
        return jnp.full_like(x_t, 0.1)
    trajectory = ddim_trajectory(initial, levels, betas, predictor)
    alpha_bar = jnp.concatenate((jnp.ones((1,)), jnp.cumprod(1.0 - betas)))
    current = initial
    expected = [initial]
    for t, s in ((4, 2), (2, 0)):
        epsilon = jnp.full_like(current, 0.1)
        x0_pred = (current - jnp.sqrt(1.0 - alpha_bar[t]) * epsilon) / jnp.sqrt(alpha_bar[t])
        current = jnp.sqrt(alpha_bar[s]) * x0_pred + jnp.sqrt(1.0 - alpha_bar[s]) * epsilon
        expected.append(current)
    np.testing.assert_allclose(np.asarray(trajectory), np.asarray(jnp.stack(expected)),
                               rtol=1e-6, atol=1e-6)
    repeated = ddim_trajectory(initial, levels, betas, predictor)
    np.testing.assert_array_equal(np.asarray(trajectory), np.asarray(repeated))
    final = ddim_sample(initial, levels, betas, predictor)
    np.testing.assert_array_equal(np.asarray(final), np.asarray(trajectory[-1]))


def test_mixture_conditional_mean_noise_respects_symmetry_and_component_mean():
    betas = beta_schedule()
    zero = jnp.zeros((2, 2), dtype=jnp.float32)
    for level in (1, 100, 1000):
        prediction = mixture_conditional_mean_noise(zero, level, betas)
        np.testing.assert_allclose(np.asarray(prediction), 0.0, atol=1e-6)

    level = 10
    alpha_bar = jnp.prod(1.0 - betas[:level])
    component_mean = jnp.sqrt(alpha_bar) * CENTERS[:1]
    prediction = mixture_conditional_mean_noise(component_mean, level, betas)
    np.testing.assert_allclose(np.asarray(prediction), 0.0, atol=1e-5)


def test_mode_metrics_report_nearest_assignment_population_and_spread():
    centers = np.array([[3.0, 0.0], [-3.0, 0.0]])
    samples = np.array([[2.8, -0.1], [3.2, 0.1], [-3.0, 0.2]])
    labels = assign_nearest_mode(samples, centers)
    np.testing.assert_array_equal(labels, [0, 0, 1])
    results = summarize_modes(samples, centers, labels)
    assert results[0]["population_count"] == 2
    assert results[0]["population_fraction"] == pytest.approx(2 / 3)
    np.testing.assert_allclose(results[0]["empirical_center"], [3.0, 0.0])
    np.testing.assert_allclose(results[0]["coordinate_variance"], [0.04, 0.01])
    np.testing.assert_allclose(results[0]["coordinate_std"], [0.2, 0.1])
    assert results[1]["population_count"] == 1


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

    terminal = jnp.array([[0.25, -0.5]], dtype=jnp.float32)
    fixed_noise = jnp.arange(6, dtype=jnp.float32).reshape(3, 1, 2) / 10
    path = ddpm_trajectory(params, terminal, fixed_noise, betas, time_features=4)
    assert path.shape == (4, 1, 2)  # x_T through x_0.
    manual = terminal
    for scan_index, level_index in enumerate((2, 1, 0)):
        beta = betas[level_index]
        manual = manual / jnp.sqrt(1.0 - beta)
        if level_index > 0:
            manual = manual + jnp.sqrt(beta) * fixed_noise[scan_index]
    np.testing.assert_allclose(np.asarray(path[-1]), np.asarray(manual), rtol=1e-6, atol=1e-6)
