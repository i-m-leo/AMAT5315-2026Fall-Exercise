#!/usr/bin/env python3
"""Part 5 verification (1): the transformer against every count model.

Plots held-out bits per character of the uniform, unigram, bigram, trigram,
4-gram, 5-gram, 6-gram and transformer models on one chart, and samples a
60-token story opening from each at temperature 1 with the same seed, shown in
order of bits per character.  Attention maps for one held-out story follow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import ngram
import samplers
from tinystories import ARTIFACTS, WEEK, load_split
from transformer import CONTEXT, forward

EVIDENCE = WEEK / "evidence"
SAMPLE_TOKENS = 60
SAMPLE_SEED = 1848
PROMPT = "Once upon a time"


def transformer_logits(params, tokens: np.ndarray) -> np.ndarray:
    window = tokens[-CONTEXT:][None, :]
    return np.asarray(forward(params, jnp.asarray(window))[0, -1])


def open_story(tokenizer, count_set: ngram.CountSet, order: int, smoothing: float,
               params, rng: np.random.Generator) -> tuple[str, list[int]]:
    """Draw 60 tokens at temperature 1, after the fixed prompt."""
    prompt = tokenizer.encode(PROMPT)
    tokens = list(prompt)
    for _ in range(SAMPLE_TOKENS):
        if order == 0:
            probability = np.full(count_set.vocabulary_size, 1.0 / count_set.vocabulary_size)
        elif order > 0:
            key = 0
            history = prompt if len(prompt) >= order - 1 else [ngram.EOS_ID] * (order - 1)
            if order > 1:
                history = tokens[-(order - 1):] if len(tokens) >= order - 1 else history
            for token in history[-(order - 1):] if order > 1 else []:
                key = key * count_set.vocabulary_size + int(token)
            probability = count_set.distribution(order, key, smoothing)
        token = int(samplers.draw(rng, probability, 1)[0])
        tokens.append(token)
    text = tokenizer.decode(tokens[len(prompt):])
    return text, tokens


def transformer_open_story(tokenizer, params, rng: np.random.Generator) -> str:
    tokens = list(tokenizer.encode(PROMPT))
    for _ in range(SAMPLE_TOKENS):
        logits = transformer_logits(params, np.asarray(tokens))
        tokens.append(int(samplers.draw(rng, samplers.softmax(logits), 1)[0]))
    return tokenizer.decode(tokens[len(tokenizer.encode(PROMPT)):])


def attention_maps(params, tokens: np.ndarray) -> np.ndarray:
    """Attention weights of shape (layers, heads, length, length)."""
    _, attentions = forward(params, jnp.asarray(tokens[None, :]), return_attention=True)
    return np.asarray(np.stack([np.asarray(a)[0] for a in attentions]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=ARTIFACTS / "transformer")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "model-comparison.json")
    args = parser.parse_args()

    from train_transformer import unflatten

    split = load_split()
    train, valid = split["train"], split["valid"]
    characters_per_token = split["stats"]["valid"]["characters_per_token"]
    ladder = json.loads((ARTIFACTS / "ngram-ladder.json").read_text())
    params = unflatten(np.load(args.parameters / "parameters.npz"))
    metrics = json.loads((args.parameters / "metrics.json").read_text())
    tokenizer = __import__("tinystories").load_tokenizer()

    rows = [{"name": "uniform", "order": 0, "smoothing": None,
             "bits_per_character": ladder["uniform"]["bits_per_character"],
             "nats_per_token": ladder["uniform"]["nats_per_token"]}]
    for entry in ladder["orders"]:
        rows.append({"name": {1: "unigram", 2: "bigram", 3: "trigram", 4: "4-gram",
                              5: "5-gram", 6: "6-gram"}[entry["n"]],
                     "order": entry["n"], "smoothing": entry["smoothing"],
                     "bits_per_character": entry["bits_per_character"],
                     "nats_per_token": entry["nats_per_token"]})
    rows.append({"name": "transformer", "order": None, "smoothing": None,
                 "bits_per_character": metrics["held_out_bits_per_character"],
                 "nats_per_token": metrics["held_out_nats"]})

    count_set = ngram.CountSet(train, split["stats"]["vocabulary_size"])
    rng = np.random.default_rng(SAMPLE_SEED)
    for row in rows:
        if row["name"] == "transformer":
            row["sample"] = transformer_open_story(tokenizer, params, rng)
        else:
            row["sample"], _ = open_story(tokenizer, count_set, row["order"],
                                          row["smoothing"], params, rng)
        print(f"{row['name']:12s} {row['bits_per_character']:.3f} bpc  {row['sample'][:70]!r}")

    plotted = rows[:4] + [rows[4], rows[-1]]
    figure, axis = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(8.0, 4.2))
    names = [row["name"] for row in plotted]
    values = [row["bits_per_character"] for row in plotted]
    colours = ["#a0aec0", "#a0aec0", "#a0aec0", "#a0aec0", "#4a5568", "#2b6cb0"]
    axis.bar(names, values, color=colours)
    for index, value in enumerate(values):
        axis.annotate(f"{value:.3f}", (index, value), ha="center", va="bottom", fontsize=8)
    axis.axhline(rows[4]["bits_per_character"], color="#c53030", ls="--", lw=1.0)
    axis.set_ylabel("held-out bits per character")
    axis.set_title("Count models and the transformer on the same held-out tokens")
    axis.grid(alpha=0.25, axis="y")
    figure.tight_layout()
    figure.savefig(EVIDENCE / "model-comparison.png", dpi=160, bbox_inches="tight")
    __import__("matplotlib.pyplot", fromlist=["pyplot"]).close(figure)

    story = valid[:CONTEXT]
    maps = attention_maps(params, story)
    figure, axes = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(
        2, 4, figsize=(13.0, 6.4))
    labels = [tokenizer.decode([int(i)]).replace(" ", "\u00b7") for i in story]
    for layer in range(maps.shape[0]):
        for head in range(maps.shape[1]):
            axis = axes[layer, head]
            image = axis.imshow(maps[layer, head], cmap="Blues", aspect="auto",
                                vmin=0.0, vmax=1.0)
            axis.set_title(f"layer {layer + 1}, head {head + 1}", fontsize=8)
            if layer == 1:
                axis.set_xticks(range(0, CONTEXT, 4))
                axis.set_xticklabels([labels[i] for i in range(0, CONTEXT, 4)],
                                     rotation=90, fontsize=6)
                axis.set_xlabel("read", fontsize=8)
            else:
                axis.set_xticks([])
            if head == 0:
                axis.set_ylabel("predicting", fontsize=8)
            axis.set_yticks([])
    figure.suptitle("Attention weights of the eight heads on the opening of a "
                    "held-out story", fontsize=10)
    figure.tight_layout()
    figure.savefig(EVIDENCE / "transformer-attention-heads.png", dpi=150, bbox_inches="tight")

    result = {"models": rows, "sample_tokens": SAMPLE_TOKENS, "seed": SAMPLE_SEED,
              "best_count_model": min(rows[:-1], key=lambda r: r["bits_per_character"])["name"],
              "figures": ["evidence/model-comparison.png",
                          "evidence/transformer-attention-heads.png"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
