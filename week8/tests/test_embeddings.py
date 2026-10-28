"""Embedding geometry: offsets, the unit-norm step and the shuffled control."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import embeddings


def synthetic_table():
    """A table where every pair has exactly the same offset.

    All the key words share one row and all the value words share another, so
    each pair's offset is the same vector and the relation cosines must be one.
    """
    table = np.zeros((40, 8))
    words = {}
    for index in range(20):
        words[f"k{index}"] = 2 * index
        words[f"v{index}"] = 2 * index + 1
        table[2 * index] = np.eye(8)[0]
        table[2 * index + 1] = np.eye(8)[1]
    return table, words


def test_unit_rows_have_length_one():
    table, _ = synthetic_table()
    scaled = embeddings.unit(table)
    np.testing.assert_allclose(np.linalg.norm(scaled, axis=1), 1.0)


def test_relation_cosines_are_high_for_parallel_offsets():
    table, words = synthetic_table()
    pairs = [(f"k{i}", f"v{i}") for i in range(9)]
    cosines = embeddings.relation_cosines(table, words, pairs)
    np.testing.assert_allclose(cosines, 1.0, atol=1e-12)


def test_relation_cosines_are_near_zero_for_random_offsets():
    rng = np.random.default_rng(3)
    table = rng.normal(size=(40, 32))
    words = {f"w{index}": index for index in range(40)}
    pairs = [(f"w{2 * i}", f"w{2 * i + 1}") for i in range(9)]
    cosines = embeddings.relation_cosines(table, words, pairs)
    assert abs(cosines.mean()) < 0.3


def test_missing_words_are_reported():
    table, words = synthetic_table()
    with pytest.raises(KeyError, match="nowhere"):
        embeddings.relation_cosines(table, words, [("k0", "nowhere")])


def test_the_three_relations_have_nine_pairs_each():
    assert len(embeddings.RELATIONS) == 3
    for pairs in embeddings.RELATIONS.values():
        assert len(pairs) == 9
        assert all(len(pair) == 2 for pair in pairs)


def test_shuffling_preserves_the_vectors_but_not_the_words():
    table, words = synthetic_table()
    rng = np.random.default_rng(embeddings.SHUFFLE_SEED)
    control = table[rng.permutation(len(table))]
    assert control.shape == table.shape
    np.testing.assert_allclose(np.sort(control.ravel()), np.sort(table.ravel()))
    np.testing.assert_allclose(np.sort(np.linalg.norm(control, axis=1)),
                               np.sort(np.linalg.norm(table, axis=1)))
