"""The transformer: shapes, masking, normalization and the training schedule."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train_transformer
import transformer


def small_params(vocabulary=11):
    return transformer.init_params(jax.random.PRNGKey(0), vocabulary)


def test_layer_norm_makes_each_row_mean_zero_variance_one():
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 7)) * 4.0 + 2.0
    gain = jnp.ones(7)
    bias = jnp.zeros(7)
    normalized = transformer.layer_norm(x, gain, bias, epsilon=0.0)
    np.testing.assert_allclose(np.asarray(normalized.mean(axis=-1)), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(normalized.var(axis=-1)), 1.0, atol=1e-5)


def test_causal_mask_gives_future_positions_exactly_zero_weight():
    tokens = jnp.asarray(np.random.default_rng(2).integers(0, 11, (2, 9)))
    _, attentions = transformer.forward(small_params(), tokens, return_attention=True)
    for weights in attentions:
        upper = np.asarray(weights)[:, :, np.triu_indices(9, k=1)[0], np.triu_indices(9, k=1)[1]]
        assert np.all(upper == 0.0)
        np.testing.assert_allclose(np.asarray(weights).sum(axis=-1), 1.0, atol=1e-5)


def test_forward_returns_one_logit_per_token_and_position():
    tokens = jnp.asarray(np.random.default_rng(3).integers(0, 11, (4, 16)))
    logits = transformer.forward(small_params(), tokens)
    assert logits.shape == (4, 16, 11)


def test_position_vectors_change_the_same_token_at_different_positions():
    tokens = jnp.asarray([[4, 4, 4, 4]])
    logits = np.asarray(transformer.forward(small_params(), tokens))
    assert not np.allclose(logits[0, 0], logits[0, 1])


def test_a_longer_window_than_the_context_is_rejected():
    tokens = jnp.zeros((1, transformer.CONTEXT + 1), dtype=jnp.int32)
    with pytest.raises(ValueError, match="context"):
        transformer.forward(small_params(), tokens)


def test_learning_rate_warms_up_then_decays_to_a_tenth():
    peak, warmup, total = 0.003, 10, 100
    values = [float(transformer.learning_rate(step, peak, warmup, total))
              for step in (0, 5, 9, 50, 99)]
    assert values[0] < values[2] <= peak * 1.001
    assert values[-1] == pytest.approx(peak * 0.1, rel=0.05)
    assert values[3] > values[4]


def test_loss_matches_a_manual_cross_entropy():
    params = small_params()
    tokens = jnp.asarray([[1, 2, 3, 4]])
    logits = transformer.forward(params, tokens[:, :-1])
    probability = jax.nn.log_softmax(logits, axis=-1)
    expected = -np.mean([
        float(probability[0, position, int(tokens[0, position + 1])])
        for position in range(tokens.shape[1] - 1)])
    assert float(transformer.loss_fn(params, tokens)) == pytest.approx(expected, abs=1e-5)


def test_window_batch_returns_consecutive_runs():
    tokens = np.arange(1000, dtype=np.int32)
    batch = train_transformer.window_batch(tokens, 5, 8, np.random.default_rng(0))
    assert batch.shape == (5, 9)
    for row in batch:
        np.testing.assert_array_equal(np.diff(row), 1)


def test_flatten_and_unflatten_round_trip():
    params = small_params()
    flat = {name: np.asarray(value) for name, value in train_transformer.flatten(params).items()}
    restored = train_transformer.unflatten(flat)
    assert len(restored["blocks"]) == transformer.LAYERS
    assert restored["blocks"][0]["wq"][0].shape == (transformer.WIDTH, transformer.WIDTH)
    np.testing.assert_allclose(np.asarray(restored["embedding"]),
                               np.asarray(params["embedding"]))
