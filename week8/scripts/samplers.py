#!/usr/bin/env python3
"""Part 5: next-token samplers, each drawing from its stated distribution.

``temperature`` divides the logits by ``T``; ``top_k`` keeps the k most likely
tokens; ``top_p`` keeps the most likely tokens until their total probability
reaches the threshold; ``min_p`` keeps every token whose probability is at
least a fixed fraction of the largest.  Dropped tokens are given zero
probability and the rest is renormalised.  Every function returns a full
probability vector over the vocabulary, so a sampler's exact target
distribution can be compared with the counts it produces.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def temperature_distribution(logits: np.ndarray, temperature: float) -> np.ndarray:
    """``softmax(logits / T)``; ``T -> 0`` becomes greedy."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def top_k_distribution(logits: np.ndarray, k: int,
                       temperature: float = 1.0) -> np.ndarray:
    """Keep the ``k`` most likely tokens, then renormalise the rest."""
    probability = temperature_distribution(logits, temperature)
    if not 1 <= k <= len(probability):
        raise ValueError("k must lie between 1 and the vocabulary size")
    keep = np.zeros(len(probability), dtype=bool)
    keep[np.argsort(probability)[::-1][:k]] = True
    trimmed = np.where(keep, probability, 0.0)
    return trimmed / trimmed.sum()


def top_p_distribution(logits: np.ndarray, threshold: float,
                       temperature: float = 1.0) -> np.ndarray:
    """Nucleus sampling: keep tokens until the cumulative mass reaches ``p``.

    The token that crosses the threshold is kept, which is what makes the
    sampler draw from this exact distribution; dropping it instead is the
    wrong sampler of the sheet.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    probability = temperature_distribution(logits, temperature)
    order = np.argsort(probability)[::-1]
    cumulative = np.cumsum(probability[order])
    cutoff = int(np.searchsorted(cumulative, threshold) + 1)
    cutoff = min(cutoff, len(probability))
    keep = np.zeros(len(probability), dtype=bool)
    keep[order[:cutoff]] = True
    trimmed = np.where(keep, probability, 0.0)
    return trimmed / trimmed.sum()


def min_p_distribution(logits: np.ndarray, fraction: float,
                       temperature: float = 1.0) -> np.ndarray:
    """Keep every token with probability at least ``fraction`` of the largest."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    probability = temperature_distribution(logits, temperature)
    keep = probability >= fraction * probability.max()
    trimmed = np.where(keep, probability, 0.0)
    return trimmed / trimmed.sum()


def draw(rng: np.random.Generator, probability: np.ndarray,
         count: int = 1) -> np.ndarray:
    """Draw token ids from one explicit distribution."""
    cumulative = np.cumsum(probability, dtype=np.float64)
    cumulative /= cumulative[-1]
    return np.searchsorted(cumulative, rng.random(count)).astype(np.int64)


def sample_sequence(model_logits, start: np.ndarray, num_tokens: int,
                    rng: np.random.Generator, sampler=None) -> np.ndarray:
    """Ancestral sampling: draw one token, append it, repeat.

    ``model_logits(context)`` turns the tokens so far into next-token logits;
    ``sampler(logits)`` turns those into a distribution.
    """
    tokens = list(np.asarray(start, dtype=np.int64))
    for _ in range(num_tokens):
        logits = np.asarray(model_logits(np.asarray(tokens)))
        probability = sampler(logits) if sampler else softmax(logits)
        tokens.append(int(draw(rng, probability, 1)[0]))
    return np.asarray(tokens, dtype=np.int64)


def chi_squared_against(distribution: np.ndarray, counts: np.ndarray,
                        minimum_expected: float = 5.0):
    """Chi-squared statistic, degrees of freedom and p-value for drawn counts.

    Rare tokens are pooled into one bin so every expected count is at least
    ``minimum_expected``.
    """
    from scipy import stats

    expected = distribution * counts.sum()
    keep = expected >= minimum_expected
    statistic = float(((counts[keep] - expected[keep]) ** 2 / expected[keep]).sum())
    degrees = int(keep.sum()) - 1
    rest_expected = expected[~keep].sum()
    if rest_expected > 0.0:
        statistic += float((counts[~keep].sum() - rest_expected) ** 2 / rest_expected)
        degrees += 1
    return statistic, degrees, float(stats.chi2.sf(statistic, degrees))
