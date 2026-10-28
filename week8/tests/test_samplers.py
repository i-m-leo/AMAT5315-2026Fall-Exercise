"""The samplers draw from the distributions they claim."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import samplers


def broad_logits():
    rng = np.random.default_rng(5)
    return rng.normal(scale=1.5, size=200)


def test_temperature_divides_the_logits():
    logits = np.asarray([1.0, 2.0, 3.0])
    hot = samplers.temperature_distribution(logits, 2.0)
    cold = samplers.temperature_distribution(logits, 0.5)
    np.testing.assert_allclose(hot, samplers.softmax(logits / 2.0))
    np.testing.assert_allclose(cold, samplers.softmax(logits / 0.5))
    assert cold.max() > hot.max()


def test_every_sampler_returns_a_distribution():
    logits = broad_logits()
    targets = [
        samplers.temperature_distribution(logits, 0.8),
        samplers.top_k_distribution(logits, 20, 0.8),
        samplers.top_p_distribution(logits, 0.9, 0.8),
        samplers.min_p_distribution(logits, 0.1, 0.7),
    ]
    for probability in targets:
        assert np.all(probability >= 0.0)
        np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-12)


def test_top_k_keeps_exactly_k_tokens():
    logits = broad_logits()
    probability = samplers.top_k_distribution(logits, 5)
    assert int((probability > 0).sum()) == 5
    assert set(np.flatnonzero(probability)) == set(np.argsort(logits)[::-1][:5])


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    logits = broad_logits()
    probability = samplers.softmax(logits)
    order = np.argsort(probability)[::-1]
    cumulative = np.cumsum(probability[order])
    cutoff = int(np.searchsorted(cumulative, 0.9) + 1)
    kept = samplers.top_p_distribution(logits, 0.9)
    assert int((kept > 0).sum()) == cutoff


def test_min_p_keeps_tokens_above_the_fraction_of_the_mode():
    logits = broad_logits()
    probability = samplers.softmax(logits)
    kept = samplers.min_p_distribution(logits, 0.2)
    assert set(np.flatnonzero(kept)) == set(np.flatnonzero(probability >= 0.2 * probability.max()))


def test_tiny_temperature_is_greedy():
    logits = broad_logits()
    probability = samplers.temperature_distribution(logits, 1e-4)
    np.testing.assert_array_equal(np.flatnonzero(probability), [int(np.argmax(logits))])


def test_draws_match_the_target_distribution():
    logits = broad_logits()
    rng = np.random.default_rng(7)
    for target in (samplers.temperature_distribution(logits, 0.8),
                   samplers.top_p_distribution(logits, 0.9, 0.8),
                   samplers.min_p_distribution(logits, 0.1, 0.7)):
        counts = np.bincount(samplers.draw(rng, target, 100_000),
                             minlength=len(target)).astype(np.float64)
        _, _, p_value = samplers.chi_squared_against(target, counts)
        assert p_value > 0.001


def test_invalid_sampler_settings_are_rejected():
    logits = broad_logits()
    with pytest.raises(ValueError):
        samplers.temperature_distribution(logits, 0.0)
    with pytest.raises(ValueError):
        samplers.top_k_distribution(logits, 0)
    with pytest.raises(ValueError):
        samplers.top_p_distribution(logits, 1.5)
    with pytest.raises(ValueError):
        samplers.min_p_distribution(logits, 0.0)
