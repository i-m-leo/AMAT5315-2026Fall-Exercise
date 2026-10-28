"""Count models: normalization, smoothing, sampling and the unseen fraction."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import ngram


def tiny_counts():
    tokens = np.asarray([2, 5, 5, 7, 2, 5, 6, 2], dtype=np.int64)
    return ngram.CountSet(tokens, vocabulary_size=10, orders=range(1, 4))


def test_context_keys_pack_windows_base_v():
    tokens = np.asarray([3, 4, 5], dtype=np.int64)
    np.testing.assert_array_equal(ngram.context_keys(tokens, 2, 10), [34, 45])
    np.testing.assert_array_equal(ngram.context_keys(tokens, 0, 10), [0, 0, 0])
    with pytest.raises(ValueError):
        ngram.context_keys(tokens, 4, 10)


def test_every_ngram_row_is_normalized():
    counts = tiny_counts()
    for order in counts.orders:
        for key in np.unique(counts.tables[order]["row_keys"])[:5]:
            total = counts.distribution(order, int(key), 0.1).sum()
            np.testing.assert_allclose(total, 1.0, atol=1e-12)


def test_bigram_probabilities_are_rows_of_the_transition_matrix():
    counts = tiny_counts()
    # The stream holds three pairs that start at 5: (5,5), (5,7) and (5,6), so
    # the row of 5 spreads a third of its mass on each.
    probability = counts.distribution(2, 5, 1e-9)
    expected = np.zeros(10)
    expected[[5, 6, 7]] = 1.0 / 3.0
    np.testing.assert_allclose(probability, expected, atol=1e-8)

    # probabilities() reports the model's probability of each observed token.
    # The first 5 follows the end-of-sequence context, whose row holds two (EOS,
    # 5) pairs and one (EOS, 6); the second 5 follows 5, whose row puts a third
    # of its mass on 5.
    stream = np.asarray([5, 5], dtype=np.int64)
    np.testing.assert_allclose(counts.probabilities(2, stream, 1e-9),
                               [2.0 / 3.0, 1.0 / 3.0], atol=1e-8)


def test_zero_counts_still_get_positive_probability():
    counts = tiny_counts()
    probability = counts.distribution(3, 0, 1e-3)  # an unseen context
    assert np.all(probability > 0.0)
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-12)


def test_unseen_fraction_rises_with_the_order():
    counts = tiny_counts()
    held_out = np.asarray([9, 9, 8, 9, 8, 9], dtype=np.int64)
    fractions = [counts.unseen_fraction(order, held_out) for order in (1, 2, 3)]
    assert fractions[0] <= fractions[1] <= fractions[2]


def test_longer_context_never_increases_the_training_nll():
    """A longer order fits its own training stream at least as well."""
    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 6, size=400).astype(np.int64)
    counts = ngram.CountSet(tokens, 6, orders=range(1, 5))
    values = [counts.nll(order, tokens, 0.01).mean() for order in range(1, 5)]
    assert values[1] < values[0]
    assert values[2] <= values[1] + 1e-9


def test_choose_smoothing_returns_a_candidate_with_its_scores():
    counts = tiny_counts()
    best, scores = ngram.choose_smoothing(counts, 2, np.asarray([5, 7, 2], dtype=np.int64))
    assert best in ngram.SMOOTHING_CONSTANTS
    assert set(scores) == {float(k) for k in ngram.SMOOTHING_CONSTANTS}
    assert scores[best] == min(scores.values())


def test_sample_from_distribution_follows_the_probabilities():
    rng = np.random.default_rng(11)
    probability = np.asarray([0.05, 0.15, 0.3, 0.5])
    draws = ngram.sample_from_distribution(rng, probability, 200_000)
    observed = np.bincount(draws, minlength=4) / 200_000
    np.testing.assert_allclose(observed, probability, atol=0.005)


def test_bits_per_character_converts_nats():
    np.testing.assert_allclose(ngram.bits_per_character(np.log(2.0) * 7.0, 7.0), 1.0)
