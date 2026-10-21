from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from cifar_sampling import ddpm_sample_with_snapshots, linear_betas, repaint_completion
from cifar_unet import _check_config, _group_norm, load_cifar_unet, make_parameterized_predictor


def test_linear_betas_matches_week7_ddpm_schedule():
    betas = linear_betas(1000)
    assert betas.dtype == np.float32
    assert betas.shape == (1000,)
    np.testing.assert_allclose(betas[[0, -1]], [0.0001, 0.02], rtol=1e-6)
    assert np.all(np.diff(betas) > 0)


def test_cifar_config_validation_rejects_unimplemented_architecture():
    with pytest.raises(ValueError, match="unsupported CIFAR U-Net config"):
        _check_config({"block_out_channels": [64]})


def test_group_norm_matches_per_example_channel_groups():
    x = jnp.arange(2 * 3 * 4 * 32, dtype=jnp.float32).reshape(2, 3, 4, 32) / 19.0
    params = {
        "norm.weight": jnp.linspace(0.5, 1.5, 32),
        "norm.bias": jnp.linspace(-0.2, 0.2, 32),
    }
    actual = np.asarray(_group_norm(x, params, "norm", groups=32, eps=1e-6))
    source = np.asarray(x)
    means = source.mean(axis=(1, 2), keepdims=True)
    variances = source.var(axis=(1, 2), keepdims=True)
    expected = (source - means) / np.sqrt(variances + 1e-6)
    expected = expected * np.asarray(params["norm.weight"])[None, None, None, :]
    expected += np.asarray(params["norm.bias"])[None, None, None, :]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_ddpm_uses_zero_based_model_indices_and_saves_requested_levels():
    calls = []

    def zero_predictor(_params, x, model_index):
        calls.append(np.asarray(model_index).copy())
        return jnp.zeros_like(x)

    betas = linear_betas(4)
    initial = jnp.ones((1, 3, 32, 32), dtype=jnp.float32) * 2.0
    final, states = ddpm_sample_with_snapshots(
        jax.random.PRNGKey(8), initial, betas, zero_predictor, None,
        snapshot_levels=(4, 2, 0),
    )
    np.testing.assert_array_equal(np.concatenate(calls), [3, 2, 1, 0])
    assert set(states) == {4, 2, 0}
    np.testing.assert_array_equal(states[4], np.asarray(initial))
    np.testing.assert_allclose(states[0], np.asarray(final))
    assert np.max(np.abs(np.asarray(final))) > 1.0  # Sampling state is not clipped.


def test_repaint_uses_t_minus_one_and_preserves_visible_pixels():
    calls = []

    def zero_predictor(_params, x, model_index):
        calls.extend(np.asarray(model_index).tolist())
        return jnp.zeros_like(x)

    truth = jnp.full((1, 3, 32, 32), 0.25, dtype=jnp.float32)
    mask = jnp.ones_like(truth).at[:, :, 8:24, 8:24].set(0.0)
    observed = truth * mask
    completion, evaluations = repaint_completion(
        jax.random.PRNGKey(7201), observed, mask, linear_betas(3),
        zero_predictor, None, levels=(3, 2, 1, 0),
    )
    assert evaluations == 5  # Two attempts per jump except the final jump.
    assert calls == [2, 2, 1, 1, 0]
    completed = np.asarray(completion)
    np.testing.assert_array_equal(completed[np.asarray(mask) == 1], np.asarray(truth)[np.asarray(mask) == 1])
    assert np.max(np.abs(completed[np.asarray(mask) == 0] - 0.25)) > 0.01


@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "inputs/cifar-model/diffusion_pytorch_model.safetensors").exists(),
    reason="download the pinned CIFAR-10 checkpoint to run this integration test",
)
def test_pinned_pretrained_cifar_unet_loads_and_predicts_float32():
    model_dir = Path(__file__).parents[1] / "inputs/cifar-model"
    params, config = load_cifar_unet(model_dir)
    assert config["sample_size"] == 32
    assert len(params) == 328
    predictor = make_parameterized_predictor()
    output = predictor(params, jnp.zeros((1, 3, 32, 32), dtype=jnp.float32), jnp.array([999]))
    assert output.shape == (1, 3, 32, 32)
    assert output.dtype == jnp.float32
    assert np.isfinite(np.asarray(output)).all()
