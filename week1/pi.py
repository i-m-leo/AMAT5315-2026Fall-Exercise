"""Estimate pi with a repeatable Monte Carlo simulation."""

import random


def estimate_pi(n, seed):
    """Return a seeded dart-throwing estimate of pi using ``n`` samples."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = random.Random(seed)
    inside_circle = 0

    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside_circle += 1

    return 4 * inside_circle / n
