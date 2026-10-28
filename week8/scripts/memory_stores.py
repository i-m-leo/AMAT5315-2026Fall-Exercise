#!/usr/bin/env python3
"""Part 3a: key-value memory stores and their capacity.

Each store holds ``M`` memories, a random unit key and a random unit value in
``d`` dimensions, and answers a query ``x`` with a weighted sum of the values,
``f(x) = sum_i sigma(k_i . x) v_i``.  The weight function distinguishes the
stores:

* outer product: ``sigma(s) = s``   -- one matrix ``sum_i v_i k_i^T``;
* least squares:  ``sigma(s) = s`` with ``W`` solving ``min sum |W k_i - v_i|^2``;
* ReLU powers:    ``sigma(s) = max(0, s)**p`` for ``p = 1, 2, 3``.

The interference-to-signal ratio ``R`` divides the length of the leak by the
length of the wanted value ``v_j``.  Its capacity is the ``M`` at which ``R``
crosses one, interpolated on log-log axes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tinystories import ARTIFACTS, WEEK

EVIDENCE = WEEK / "evidence"
DRAWS = 6
QUERIES = 32
CAPACITY_DIMENSIONS = (16, 32, 64)
CAPACITY_MEMORIES = [2 ** k for k in range(3, 17)]  # 8 .. 65536
RATIO_DIMENSION = 64
RATIO_MEMORIES = [2 ** k for k in range(3, 17)]
POWERS = (1, 2, 3)


def random_memories(rng: np.random.Generator, num_memories: int, dimension: int):
    """M unit keys, M unit values, and a query equal to a stored key index."""
    keys = rng.normal(size=(num_memories, dimension))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.normal(size=(num_memories, dimension))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return keys, values


def outer_product_store(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The one-matrix store ``W = sum_i v_i k_i^T``."""
    return values.T @ keys


def least_squares_store(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The matrix minimising ``sum_i |W k_i - v_i|^2``.

    With ``K`` the rows of keys, this is ``W = V^T (K^T)^+``, the minimum-norm
    solution, so it answers every stored key exactly while ``M <= d``.
    """
    return values.T @ np.linalg.pinv(keys.T)


def recall(matrix_store: np.ndarray | None, keys: np.ndarray, values: np.ndarray,
           query_index: int, power: int | None = None,
           sharpness: float | None = None) -> np.ndarray:
    """Recalled vector for the query equal to ``keys[query_index]``."""
    query = keys[query_index]
    if matrix_store is not None:
        return matrix_store @ query
    similarities = keys @ query
    if sharpness is not None:
        weights = np.exp(sharpness * similarities)
    else:
        weights = np.maximum(0.0, similarities) ** power
    return weights @ values


def interference_to_signal(recalled: np.ndarray,
                           own_value: np.ndarray) -> float:
    """Length of the leak over the length of the wanted value.

    Write the recall in the basis ``v_1, ..., v_M``.  The values are random, so
    to first order the coefficient of the wanted value is its projection,
    ``c_j = v_j . f``, and the recall is ``c_j v_j`` plus the leak.  Dividing
    the leak by ``c_j |v_j|`` removes the overall gain, so the ratio is scale
    free.  For a matrix store this projection matters: past ``d`` memories the
    answer is a shrunken mixture of all the values, and only the coefficient of
    ``v_j`` is the signal.
    """
    coefficient = float(own_value @ recalled)
    norm = float(np.linalg.norm(own_value))
    if coefficient <= 0.0 or norm == 0.0:
        return float("inf")
    leak = recalled - coefficient * own_value
    return float(np.linalg.norm(leak) / (coefficient * norm))


def store_ratios(store: str, keys: np.ndarray, values: np.ndarray,
                 query_indices: np.ndarray) -> np.ndarray:
    """Interference-to-signal ratios of one store for several stored-key queries."""
    power = None
    matrix = None
    if store == "outer":
        matrix = outer_product_store(keys, values)
    elif store == "least-squares":
        matrix = least_squares_store(keys, values)
    elif store.startswith("relu"):
        power = int(store[-1])
    else:
        raise ValueError(f"unknown store {store}")
    ratios = np.empty(len(query_indices))
    if matrix is not None:
        for position, query_index in enumerate(query_indices):
            recalled = recall(matrix, keys, values, int(query_index))
            ratios[position] = interference_to_signal(recalled, values[query_index])
        return ratios

    # ReLU stores measure against the unit value, as "divided by the picture's
    # own weight" does in Part 4.  One query at a time keeps this O(M d).
    for position, query_index in enumerate(query_indices):
        query = keys[int(query_index)]
        weights = np.maximum(0.0, keys @ query) ** power
        own_weight = float(weights[query_index])
        if own_weight == 0.0:
            ratios[position] = float("inf")
            continue
        recalled = (weights @ values) / own_weight
        ratios[position] = float(np.linalg.norm(recalled - values[query_index]))
    return ratios


def median_ratio(store: str, num_memories: int, dimension: int,
                 draws: int = DRAWS, queries: int = QUERIES) -> float:
    """Median over independent draws and queried keys of the ratio.

    Each draw uses fresh keys and values.  Querying several stored keys per draw
    averages out the accidental alignments of single random memories, and the
    median over the pooled ratios reports the typical one.
    """
    pooled = []
    for draw in range(draws):
        rng = np.random.default_rng(1000 * draw + dimension)
        keys, values = random_memories(rng, num_memories, dimension)
        query_indices = rng.choice(num_memories, size=min(queries, num_memories),
                                   replace=False)
        pooled.append(store_ratios(store, keys, values, query_indices))
    return float(np.median(np.concatenate(pooled)))


def crossing(memories: list[int], ratios: list[float]) -> float | None:
    """The M where R crosses one, linearly interpolated on log-log axes."""
    log_m = np.log2(np.asarray(memories, dtype=np.float64))
    log_r = np.log(np.asarray(ratios, dtype=np.float64))
    for index in range(1, len(ratios)):
        if ratios[index - 1] < 1.0 <= ratios[index]:
            slope = (log_r[index] - log_r[index - 1]) / (log_m[index] - log_m[index - 1])
            return float(2.0 ** (log_m[index - 1] + (0.0 - log_r[index - 1]) / slope))
    return None


def predicted_ratios(memories: np.ndarray, dimension: int) -> dict[str, np.ndarray]:
    """The leading-order predictions of the sheet."""
    memories = np.asarray(memories, dtype=np.float64)
    return {
        "outer": np.sqrt(memories / dimension),
        "least-squares": np.sqrt(np.maximum(memories / (dimension - 1) - 1.0, 1e-12)),
        "relu1": np.sqrt(memories / (2 * dimension)),
        "relu2": np.sqrt(3 * memories / (2 * dimension ** 2)),
        "relu3": np.sqrt(15 * memories / (2 * dimension ** 3)),
    }


def fitted_slope(dimensions, capacities) -> float:
    """Least-squares slope of log capacity against log dimension."""
    x = np.log(np.asarray(dimensions, dtype=np.float64))
    y = np.log(np.asarray(capacities, dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    stores = ("outer", "least-squares", "relu1", "relu2", "relu3")

    ratios = {}
    for store in stores:
        ratios[store] = [median_ratio(store, m, RATIO_DIMENSION) for m in RATIO_MEMORIES]
        print(store, " ".join(f"{r:.3g}" for r in ratios[store]))

    ratio_crossings = {store: crossing(RATIO_MEMORIES, ratios[store]) for store in stores}

    capacities = {}
    for store in stores:
        capacities[store] = []
        for dimension in CAPACITY_DIMENSIONS:
            series = [median_ratio(store, m, dimension) for m in CAPACITY_MEMORIES]
            value = crossing(CAPACITY_MEMORIES, series)
            capacities[store].append(value)
        print(store, "capacities", capacities[store])

    slopes = {store: fitted_slope(CAPACITY_DIMENSIONS, capacities[store])
              for store in stores}
    print("slopes", slopes)

    predictions = predicted_ratios(np.asarray(RATIO_MEMORIES), RATIO_DIMENSION)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.4))
    colours = {"outer": "#2b6cb0", "least-squares": "#2f855a", "relu1": "#c05621",
               "relu2": "#6b46c1", "relu3": "#b83280"}
    memories = np.asarray(RATIO_MEMORIES)
    for store in stores:
        left.plot(memories, ratios[store], "o-", ms=3, color=colours[store],
                  label=store.replace("relu", "ReLU$^{") + "}$" if store.startswith("relu") else store)
        left.plot(memories, predictions[store], ls="--", lw=1.0, color=colours[store])
    left.axhline(1.0, color="k", lw=1.0, ls=":")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("memories M")
    left.set_ylabel("interference-to-signal ratio R")
    left.set_title("R against M at d = 64 (dashed: predictions)")
    left.grid(alpha=0.3, which="both")
    left.legend(frameon=False, fontsize=8)

    dimension_grid = np.geomspace(12, 90, 60)
    for store in stores:
        right.plot(CAPACITY_DIMENSIONS, capacities[store], "o", color=colours[store])
        anchor = np.asarray(capacities[store], dtype=np.float64)
        scale = np.exp(np.mean(np.log(anchor) - slopes[store] * np.log(CAPACITY_DIMENSIONS)))
        right.plot(dimension_grid, scale * dimension_grid ** slopes[store], "--",
                   lw=1.0, color=colours[store],
                   label=f"{store}: slope {slopes[store]:.2f}")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("dimension d")
    right.set_ylabel("capacity (M where R = 1)")
    right.set_title("Capacity grows like d^p")
    right.grid(alpha=0.3, which="both")
    right.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    path = EVIDENCE / "memory-store-capacity.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    result = {
        "draws": DRAWS,
        "ratio_dimension": RATIO_DIMENSION,
        "ratio_memories": RATIO_MEMORIES,
        "ratios": ratios,
        "ratio_crossings": ratio_crossings,
        "capacity_dimensions": list(CAPACITY_DIMENSIONS),
        "capacities": capacities,
        "capacity_slopes": slopes,
        "predictions": {key: value.tolist() for key, value in predictions.items()},
        "predicted_capacity": {
            "outer": RATIO_DIMENSION,
            "least-squares": 2 * RATIO_DIMENSION,
            "relu1": 2 * RATIO_DIMENSION,
            "relu2": 2 * RATIO_DIMENSION ** 2 / 3,
            "relu3": 2 * RATIO_DIMENSION ** 3 / 15,
        },
        "figure": str(path.relative_to(WEEK)),
    }
    output = ARTIFACTS / "memory-store-capacity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("crossings", ratio_crossings)
    print(f"saved {output} and {path}")


if __name__ == "__main__":
    main()
