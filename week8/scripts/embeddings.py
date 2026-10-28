#!/usr/bin/env python3
"""Part 1: meanings as directions in a pretrained embedding table.

Loads Model2Vec's ``potion-base-8M`` table without PyTorch, then tests three
relations of nine word pairs each.  A relation is a direction when each pair's
offset (value minus key, both unit length) points along the average offset of
the other eight pairs.  A control with rows shuffled by seed 1730 keeps every
vector but gives each word another word's vector.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

from tinystories import ARTIFACTS, DATA, WEEK

EVIDENCE = WEEK / "evidence"
TABLE = DATA / "model2vec" / "model.safetensors"
TOKENIZER_JSON = DATA / "model2vec" / "tokenizer.json"
SHUFFLE_SEED = 1730
CONTROL_DRAWS = 20

RELATIONS = {
    "country to capital": [
        ("iran", "tehran"), ("turkey", "ankara"), ("russia", "moscow"),
        ("china", "beijing"), ("france", "paris"), ("sweden", "stockholm"),
        ("japan", "tokyo"), ("peru", "lima"), ("kenya", "nairobi"),
    ],
    "male to female": [
        ("king", "queen"), ("man", "woman"), ("he", "she"),
        ("brother", "sister"), ("father", "mother"), ("son", "daughter"),
        ("uncle", "aunt"), ("boy", "girl"), ("husband", "wife"),
    ],
    "adjective to comparative": [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"),
        ("dark", "darker"), ("cold", "colder"), ("old", "older"),
        ("strong", "stronger"), ("long", "longer"), ("tall", "taller"),
    ],
}


def load_table() -> tuple[np.ndarray, dict[str, int]]:
    """Return the embedding matrix and the token-to-id vocabulary."""
    if not TABLE.exists():
        raise FileNotFoundError(
            f"missing {TABLE}; download model.safetensors and tokenizer.json "
            "from minishlab/potion-base-8M at revision bf8b056")
    with safe_open(TABLE, framework="numpy") as handle:
        embeddings = np.asarray(handle.get_tensor("embeddings"), dtype=np.float64)
    vocabulary = json.loads(TOKENIZER_JSON.read_text())["model"]["vocab"]
    return embeddings, vocabulary


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0.0, 1.0, norms)


def relation_cosines(table: np.ndarray, vocabulary: dict[str, int],
                     pairs: list[tuple[str, str]]) -> np.ndarray:
    """Cosine of each pair's offset with the average offset of the other eight."""
    missing = [word for pair in pairs for word in pair if word not in vocabulary]
    if missing:
        raise KeyError(f"words missing from the vocabulary: {missing}")
    rows = np.stack([table[vocabulary[key]] for key, _ in pairs])
    columns = np.stack([table[vocabulary[value]] for _, value in pairs])
    offsets = unit(columns) - unit(rows)

    cosines = np.empty(len(pairs))
    for index in range(len(pairs)):
        others = np.delete(offsets, index, axis=0)
        mean_offset = others.mean(axis=0)
        cosines[index] = float(
            offsets[index] @ mean_offset
            / (np.linalg.norm(offsets[index]) * np.linalg.norm(mean_offset)))
    return cosines


def run_relations(table: np.ndarray, vocabulary: dict[str, int]) -> dict:
    results = {}
    for name, pairs in RELATIONS.items():
        cosines = relation_cosines(table, vocabulary, pairs)
        results[name] = {
            "pairs": [[a, b] for a, b in pairs],
            "cosines": cosines.tolist(),
            "mean_cosine": float(cosines.mean()),
            "weakest_pair": list(pairs[int(np.argmin(cosines))]),
            "weakest_cosine": float(cosines.min()),
        }
    return results


def shuffled_means(table: np.ndarray, vocabulary: dict[str, int],
                   seed: int = SHUFFLE_SEED) -> dict:
    """Mean cosine for each relation when table rows are shuffled by ``seed``."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(table))
    control = table[permutation]
    return {name: run_relations(control, vocabulary)[name]["mean_cosine"]
            for name in RELATIONS}


def plot(table_result: dict, control_result: dict, path: Path) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    names = list(RELATIONS)
    figure, axes = plt.subplots(1, len(names), figsize=(13.5, 4.2), sharey=True)
    for axis, name in zip(axes, names):
        cosines = np.asarray(table_result[name]["cosines"])
        control = np.asarray(control_result[name]["cosines"])
        for row, values, colour in ((1.0, cosines, "#2b6cb0"), (-1.0, control, "#a0aec0")):
            for value in values:
                angle = np.arccos(np.clip(value, -1.0, 1.0))
                axis.annotate("", xy=(value, row), xytext=(0.0, row),
                              arrowprops={"arrowstyle": "-|>", "lw": 1.2,
                                          "color": colour, "alpha": 0.85})
                axis.plot([0.0], [row], marker="o", ms=2.5, color=colour)
        axis.axhline(0.0, color="k", lw=1.0)
        axis.plot([-0.25, 1.05], [0.0, 0.0], color="k", lw=1.6,
                  solid_capstyle="round")
        axis.axvline(0.0, color="#4a5568", lw=0.8, ls="--")
        axis.set_title(f"{name}\ntable mean {cosines.mean():.2f}, "
                       f"control {control.mean():+.2f}", fontsize=10)
        axis.set_xlim(-0.45, 1.05)
        axis.set_ylim(-1.5, 1.5)
        axis.set_xlabel("cosine with the mean offset of the other pairs")
        axis.set_yticks([1.0, -1.0])
        axis.set_yticklabels(["table", "shuffled\ncontrol"])
        axis.grid(alpha=0.25, axis="x")
    figure.suptitle("Relation offsets in potion-base-8M (blue, above) and its "
                    "row-shuffled control (grey, below)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    table, vocabulary = load_table()
    table_result = run_relations(table, vocabulary)
    control_result = run_relations(table[np.random.default_rng(SHUFFLE_SEED).permutation(len(table))],
                                   vocabulary)
    control_over_20 = [shuffled_means(table, vocabulary, seed=1730 + draw)
                       for draw in range(CONTROL_DRAWS)]

    for name in RELATIONS:
        print(f"{name:26s} table {table_result[name]['mean_cosine']:+.3f}  "
              f"control {control_result[name]['mean_cosine']:+.3f}  "
              f"weakest {table_result[name]['weakest_pair']} "
              f"({table_result[name]['weakest_cosine']:+.3f})")

    extremes = {name: {
        "min": float(min(draw[name] for draw in control_over_20)),
        "max": float(max(draw[name] for draw in control_over_20)),
    } for name in RELATIONS}

    path = plot(table_result, control_result, EVIDENCE / "embedding-relations.png")
    result = {
        "table": str(TABLE.relative_to(WEEK)),
        "shape": list(table.shape),
        "shuffle_seed": SHUFFLE_SEED,
        "relations": table_result,
        "control": control_result,
        "control_20_shuffles": extremes,
        "figure": str(path.relative_to(WEEK)),
    }
    output = ARTIFACTS / "embedding-relations.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"control over {CONTROL_DRAWS} shuffles: {extremes}")
    print(f"saved {output} and {path}")


if __name__ == "__main__":
    main()
