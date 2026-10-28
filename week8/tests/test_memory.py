"""Key-value memory stores: recall, interference and capacity."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import attention_recall
import memory_stores


def memories(count=8, dimension=8, seed=0):
    return memory_stores.random_memories(np.random.default_rng(seed), count, dimension)


def test_outer_product_store_answers_perpendicular_keys_exactly():
    keys = np.eye(8)
    values = np.random.default_rng(1).normal(size=(8, 8))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    store = memory_stores.outer_product_store(keys, values)
    for index in range(8):
        np.testing.assert_allclose(store @ keys[index], values[index], atol=1e-12)


def test_least_squares_store_answers_every_key_when_m_le_d():
    keys, values = memories(count=6, dimension=10)
    store = memory_stores.least_squares_store(keys, values)
    np.testing.assert_allclose(keys @ store.T, values, atol=1e-10)


def test_one_key_is_recalled_without_interference():
    for store in ("outer", "least-squares", "relu1", "relu2", "relu3"):
        keys, values = memories(count=1, dimension=16)
        assert memory_stores.store_ratios(store, keys, values, np.asarray([0]))[0] < 1e-8


def test_interference_grows_with_the_number_of_memories():
    for store in ("outer", "relu1", "relu2"):
        small = memory_stores.store_ratios(store, *memories(4, 16), np.arange(4))[0]
        large = memory_stores.store_ratios(store, *memories(64, 16), np.arange(4))[0]
        assert large > small


def test_relu_powers_silence_more_than_relu():
    keys, values = memories(count=64, dimension=16, seed=2)
    ratios = {store: memory_stores.store_ratios(store, keys, values, np.arange(8)).mean()
              for store in ("relu1", "relu2", "relu3")}
    assert ratios["relu1"] > ratios["relu2"] > ratios["relu3"]


def test_crossing_is_interpolated_between_bracketing_memories():
    value = memory_stores.crossing([8, 16, 32, 64], [0.5, 0.9, 1.5, 2.5])
    assert 16.0 < value < 32.0
    # On log-log axes, a value of one exactly at M = 16 is returned unchanged.
    assert memory_stores.crossing([8, 16, 32], [0.5, 1.0, 2.0]) == pytest.approx(16.0)
    assert memory_stores.crossing([8, 16], [0.5, 0.9]) is None


def test_fitted_slope_recovers_a_power_law():
    dimensions = [16, 32, 64]
    capacities = [3.0 * d ** 2 for d in dimensions]
    assert memory_stores.fitted_slope(dimensions, capacities) == pytest.approx(2.0)


def test_softmax_store_beats_the_outer_product_store():
    dimension = 32
    rng = np.random.default_rng(4)
    keys = attention_recall.random_unit_keys(rng, 256, dimension)
    values = rng.choice((-1.0, 1.0), size=(256, 64))
    _, outer = attention_recall.recall_all("outer", keys, values, 0)
    _, softmax = attention_recall.recall_all("softmax", keys, values, 0)
    assert softmax < outer


def test_character_picture_is_a_two_valued_square():
    picture = attention_recall.character_picture("A")
    assert picture.shape == (attention_recall.PIXELS,)
    assert set(np.unique(picture)) <= {-1.0, 1.0}
    assert (picture > 0).any()
