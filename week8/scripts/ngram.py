#!/usr/bin/env python3
"""Count models (n-gram Markov chains) on the BPE TinyStories tokens.

An n-gram model of order ``n`` keeps the last ``n - 1`` tokens as its context
and predicts the next token from counts, with add-k smoothing.  Contexts are
packed into int64 keys, ``key = x_0 * V**(n-2) + ... + x_{n-2}``, so a context
is an ordinary integer.  The whole model is then a pair of sorted arrays: one
for the row counts ``n(context)`` and one for the pair counts
``n(context, next)``.

The end-of-sequence token separates stories, so it is the context of each
story's first token.  The token stream is left-padded with ``n - 1`` copies of
EOS for the same reason; that affects only the first token.
"""
from __future__ import annotations

import numpy as np

SMOOTHING_CONSTANTS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
EOS_ID = 2


def context_keys(tokens: np.ndarray, order: int, vocabulary_size: int) -> np.ndarray:
    """Pack every length-``order`` window of ``tokens`` into one int64 key.

    Window ``t`` covers ``tokens[t : t + order]``; the result has
    ``len(tokens) - order + 1`` entries.  ``order == 0`` gives zeros.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return np.zeros(len(tokens), dtype=np.int64)
    window = len(tokens) - order + 1
    if window < 1:
        raise ValueError("tokens are shorter than one window")
    keys = np.zeros(window, dtype=np.int64)
    for offset in range(order):
        keys = keys * vocabulary_size + tokens[offset:offset + window].astype(np.int64)
    return keys


def _lookup(sorted_keys: np.ndarray, query: np.ndarray,
            values: np.ndarray) -> np.ndarray:
    """Return ``values`` at ``query`` in a key-sorted table, or zero if absent."""
    if len(sorted_keys) == 0:
        return np.zeros(query.shape, dtype=np.float64)
    position = np.searchsorted(sorted_keys, query)
    position = np.clip(position, 0, len(sorted_keys) - 1)
    return np.where(sorted_keys[position] == query, values[position], 0).astype(np.float64)


class CountSet:
    """The n-gram count tables of one training token stream, for orders 1..6."""

    def __init__(self, train_tokens: np.ndarray, vocabulary_size: int,
                 orders=range(1, 7)) -> None:
        self.vocabulary_size = int(vocabulary_size)
        self.orders = list(orders)
        self.train_tokens = np.asarray(train_tokens, dtype=np.int64)
        if self.train_tokens.size == 0:
            raise ValueError("an empty training stream has no counts")
        self.tables: dict[int, dict] = {}
        for order in self.orders:
            padded = self._pad(self.train_tokens, order)
            contexts = context_keys(padded, order - 1, self.vocabulary_size)[:len(self.train_tokens)]
            full = context_keys(padded, order, self.vocabulary_size)[:len(self.train_tokens)]
            row_keys, row_counts = np.unique(contexts, return_counts=True)
            full_keys, full_counts = np.unique(full, return_counts=True)
            pair_keys, pair_counts = np.unique(
                contexts * self.vocabulary_size + self.train_tokens, return_counts=True)
            self.tables[order] = {
                "row_keys": row_keys,
                "row_counts": row_counts.astype(np.int64),
                "full_keys": full_keys,
                "full_counts": full_counts.astype(np.int64),
                "pair_keys": pair_keys,
                "pair_counts": pair_counts.astype(np.int64),
                "contexts_seen": len(row_keys),
                "possible_contexts": self.vocabulary_size ** (order - 1),
            }

    def _pad(self, tokens: np.ndarray, order: int) -> np.ndarray:
        return np.concatenate((
            np.full(order - 1, EOS_ID, dtype=np.int64), tokens))

    def _contexts(self, tokens: np.ndarray, order: int) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        padded = self._pad(tokens, order)
        return context_keys(padded, order - 1, self.vocabulary_size)[:len(tokens)]

    def probabilities(self, order: int, tokens: np.ndarray,
                      smoothing: float) -> np.ndarray:
        """Next-token probability of every token, given the stream before it."""
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        table = self.tables[order]
        contexts = self._contexts(tokens, order)
        pairs = contexts * self.vocabulary_size + np.asarray(tokens, dtype=np.int64)
        row = _lookup(table["row_keys"], contexts, table["row_counts"])
        pair = _lookup(table["pair_keys"], pairs, table["pair_counts"])
        return (pair + smoothing) / (row + smoothing * self.vocabulary_size)

    def nll(self, order: int, tokens: np.ndarray, smoothing: float) -> np.ndarray:
        """Per-token negative log probability, in nats."""
        return -np.log(self.probabilities(order, tokens, smoothing))

    def unseen_fraction(self, order: int, tokens: np.ndarray) -> float:
        """Fraction of held-out n-grams absent from the training stream."""
        table = self.tables[order]
        tokens = np.asarray(tokens, dtype=np.int64)
        padded = self._pad(tokens, order)
        full = context_keys(padded, order, self.vocabulary_size)[:len(tokens)]
        seen = _lookup(table["full_keys"], full, table["full_counts"])
        return float(np.mean(seen == 0))

    def distribution(self, order: int, context_key: int,
                     smoothing: float) -> np.ndarray:
        """The full next-token distribution after one packed context key."""
        table = self.tables[order]
        if order == 1:
            row_count = float(table["row_counts"][0])
            counts = _lookup(table["pair_keys"],
                             np.arange(self.vocabulary_size, dtype=np.int64),
                             table["pair_counts"])
        else:
            row_count = float(_lookup(table["row_keys"],
                                      np.asarray([context_key]), table["row_counts"])[0])
            counts = _lookup(
                table["pair_keys"],
                context_key * self.vocabulary_size + np.arange(self.vocabulary_size, dtype=np.int64),
                table["pair_counts"])
        return (counts + smoothing) / (row_count + smoothing * self.vocabulary_size)

    def context_of(self, order: int, tokens: np.ndarray) -> np.ndarray:
        """Packed context key of each token in ``tokens``."""
        return self._contexts(tokens, order)


def choose_smoothing(count_set: CountSet, order: int,
                     tokens: np.ndarray) -> tuple[float, dict]:
    """Pick add-k on a held-out chunk of training tokens, never on validation."""
    scores = {float(k): float(np.mean(count_set.nll(order, tokens, k)))
              for k in SMOOTHING_CONSTANTS}
    best = min(scores, key=scores.get)
    return best, scores


def sample_from_distribution(rng: np.random.Generator,
                             probability: np.ndarray, num_draws: int) -> np.ndarray:
    """Draw token ids from one categorical distribution."""
    cumulative = np.cumsum(probability.astype(np.float64))
    cumulative /= cumulative[-1]
    return np.searchsorted(cumulative, rng.random(num_draws)).astype(np.int64)


def sample_markov_chain(rng: np.random.Generator, count_set: CountSet,
                        order: int, start: np.ndarray, num_tokens: int,
                        smoothing: float) -> np.ndarray:
    """Ancestral sampling of a Markov chain of the given order."""
    history = list(np.asarray(start, dtype=np.int64))
    for _ in range(order - 1):
        history.insert(0, EOS_ID)
    drawn = np.empty(num_tokens, dtype=np.int64)
    for step in range(num_tokens):
        key = 0
        for token in history[-(order - 1):] if order > 1 else []:
            key = key * count_set.vocabulary_size + int(token)
        probability = count_set.distribution(order, key, smoothing)
        token = int(sample_from_distribution(rng, probability, 1)[0])
        drawn[step] = token
        history.append(token)
    return drawn


def bits_per_character(total_nats: float, characters: float) -> float:
    """Convert a total held-out negative log likelihood to bits per character."""
    return total_nats / np.log(2.0) / characters
