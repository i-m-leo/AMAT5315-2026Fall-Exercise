#!/usr/bin/env python3
"""Part 4: recalling a picture from a key-value store built out of memories.

A 32x32 picture of one character is stored as one value among ``M`` memories in
``d = 64`` key dimensions; the other values are random +-1 pictures.  Querying
with a stored key returns the picture plus the leak of the other memories,
divided by the matching weight.  The outer-product, ReLU and ReLU^2 stores lose
the picture as ``M`` grows; the softmax store at sharpness ``beta = 8`` -- the
``1/sqrt(d)`` scale of attention -- keeps it clean.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tinystories import ARTIFACTS, WEEK
from memory_stores import outer_product_store

EVIDENCE = WEEK / "evidence"
DIMENSION = 64
SIDE = 32
PIXELS = SIDE * SIDE
MEMORY_COUNTS = (64, 256, 1024, 4096)
SHARPNESS = 8.0
DRAWS = 20
MANY_DRAW_MEMORIES = 1024
STORES = ("outer", "relu1", "relu2", "softmax")


def character_picture(character: str = "A", boxes: int = SIDE,
                      resolution: int = 4, font_size: float = 96.0) -> np.ndarray:
    """Render a character into a ``boxes`` x ``boxes`` array of +1 ink, -1 background.

    The character is drawn large, then each output pixel takes the darkest
    sub-pixel of a ``resolution`` x ``resolution`` block, which keeps thin
    strokes from vanishing.
    """
    side = boxes * resolution
    figure = plt.figure(figsize=(2, 2), dpi=side // 2)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.text(0.5, 0.5, character, ha="center", va="center",
              fontsize=font_size, color="black")
    figure.canvas.draw()
    grey = np.asarray(figure.canvas.buffer_rgba())[..., 0].astype(np.float64)
    plt.close(figure)
    grey = grey[-side:, :side] if grey.shape[0] >= side else grey
    blocks = grey[:side, :side].reshape(boxes, resolution, boxes, resolution)
    darkest = blocks.min(axis=(1, 3))
    return np.where(darkest < 128.0, 1.0, -1.0).reshape(-1)


def random_unit_keys(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    keys = rng.normal(size=(count, dimension))
    return keys / np.linalg.norm(keys, axis=1, keepdims=True)


def recall_all(store: str, keys: np.ndarray, values: np.ndarray,
               picture_index: int, sharpness: float = SHARPNESS):
    """Return the recalled picture and its interference-to-signal ratio."""
    query = keys[picture_index]
    similarities = keys @ query
    if store == "outer":
        recalled = (values.T @ keys) @ query
        weight = float(similarities[picture_index])
    elif store.startswith("relu"):
        power = int(store[-1])
        weights = np.maximum(0.0, similarities) ** power
        weight = float(weights[picture_index])
        recalled = weights @ values
    elif store == "softmax":
        weights = np.exp(sharpness * similarities)
        weight = float(weights[picture_index])
        recalled = weights @ values
    else:
        raise ValueError(store)
    if weight <= 0.0:
        return np.full(values.shape[1], np.nan), float("inf")
    recalled = recalled / weight
    target = values[picture_index]
    leak = np.linalg.norm(recalled - target) / np.linalg.norm(target)
    return recalled, float(leak)


def predicted_ratio(store: str, memories: int) -> float:
    """The leading-order prediction for each store at d = 64."""
    if store == "outer":
        return np.sqrt(memories / DIMENSION)
    if store == "relu1":
        return np.sqrt(memories / (2 * DIMENSION))
    if store == "relu2":
        return np.sqrt(3 * memories / (2 * DIMENSION ** 2))
    if store == "softmax":
        return np.sqrt(memories * np.exp(SHARPNESS ** 2 / DIMENSION - 2 * SHARPNESS))
    raise ValueError(store)


def main() -> None:
    picture = character_picture()
    median_ratios = {store: {} for store in STORES}
    total = {store: {} for store in STORES}
    galleries = {}

    for memories in MEMORY_COUNTS:
        rng = np.random.default_rng(memories)
        values = rng.choice((-1.0, 1.0), size=(memories, PIXELS))
        values[0] = picture
        keys = random_unit_keys(rng, memories, DIMENSION)
        galleries[memories] = {}
        for store in STORES:
            recalled, ratio = recall_all(store, keys, values, 0)
            galleries[memories][store] = recalled
            total[store][memories] = ratio
            if memories == MANY_DRAW_MEMORIES:
                draws = []
                for draw in range(DRAWS):
                    rng = np.random.default_rng(10_000 * draw + memories)
                    v = rng.choice((-1.0, 1.0), size=(memories, PIXELS))
                    v[0] = picture
                    k = random_unit_keys(rng, memories, DIMENSION)
                    draws.append(recall_all(store, k, v, 0)[1])
                median_ratios[store][memories] = float(np.median(draws))
            else:
                median_ratios[store][memories] = ratio
            # How much of the character survives: the fraction of the picture's
            # ink pixels that stay positive in the recall.
            ink = recalled[picture > 0]
            readable = float(np.mean(ink > 0))
            print(f"M={memories:5d} {store:8s} R={median_ratios[store][memories]:.3g} "
                  f"(predicted {predicted_ratio(store, memories):.3g}) "
                  f"ink kept {100 * readable:5.1f}%")

    figure, axes = plt.subplots(len(MEMORY_COUNTS), len(STORES),
                                figsize=(2.1 * len(STORES), 2.15 * len(MEMORY_COUNTS)))
    for row, memories in enumerate(MEMORY_COUNTS):
        for column, store in enumerate(STORES):
            axis = axes[row, column]
            recalled = galleries[memories][store].reshape(SIDE, SIDE)
            ratio = median_ratios[store][memories]
            grid_y, grid_x = np.mgrid[0:SIDE, 0:SIDE]
            positive = np.clip(recalled, 0.0, None)
            size = 6.0 * positive / max(positive.max(), 1e-9) + 0.05
            axis.scatter(grid_x.ravel(), grid_y.ravel(), s=size.ravel(),
                         c=np.where(recalled.ravel() > 0, "#c53030", "#cbd5e0"),
                         marker="s", linewidths=0)
            axis.set_xlim(-0.5, SIDE - 0.5)
            axis.set_ylim(SIDE - 0.5, -0.5)
            axis.set_xticks([])
            axis.set_yticks([])
            colour = "#c53030" if ratio < 1.0 else "#2d3748"
            axis.set_title(f"R = {ratio:.2f}", fontsize=8, color=colour)
            if row == 0:
                axis.annotate(store, (0.5, 1.35), xycoords="axes fraction",
                              ha="center", fontsize=9, weight="bold")
            if column == 0:
                axis.annotate(f"M = {memories}", (-0.18, 0.5), xycoords="axes fraction",
                              ha="right", va="center", rotation=90, fontsize=8)
    figure.suptitle("Recalled picture divided by its own weight, d = 64, "
                    "softmax sharpness 8\n(dot area grows with the recalled pixel; "
                    "red marks R < 1)", fontsize=10)
    figure.tight_layout()
    path = EVIDENCE / "attention-picture-recall.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    result = {
        "dimension": DIMENSION,
        "picture": SIDE,
        "sharpness": SHARPNESS,
        "memory_counts": list(MEMORY_COUNTS),
        "ratios": {store: {str(m): median_ratios[store][m] for m in MEMORY_COUNTS}
                   for store in STORES},
        "predicted": {store: {str(m): float(predicted_ratio(store, m))
                              for m in MEMORY_COUNTS} for store in STORES},
        "draws": DRAWS,
        "figure": str(path.relative_to(WEEK)),
    }
    output = ARTIFACTS / "attention-picture-recall.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {output} and {path}")


if __name__ == "__main__":
    main()
