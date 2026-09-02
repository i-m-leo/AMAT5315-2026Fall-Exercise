# Week 1 π Estimator Specification

`estimate_pi(n, seed)` estimates π by generating `n` repeatable random points in the unit square from `(0, 0)` to `(1, 1)`, counting the points whose distance from the origin is at most 1, and returning four times the fraction inside the quarter circle. The `seed` argument fixes the random sequence, so calls with the same inputs return the same result. Correctness is defined by the following assertion: `abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2`.
