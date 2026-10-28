#!/usr/bin/env python3
"""Part 3b: a one-token MLP language model, and its hidden units as memories.

The model is Bengio's: embed the previous token in 64 dimensions, apply one
hidden layer of 512 tanh units, and read out one logit per token of the
512-token vocabulary.  It sees exactly what the bigram sees, so a well-trained
model reaches the bigram's held-out loss -- a check on training, not a feature.

The verification asks for a held-out loss within 0.01 nats of the bigram's
3.038.  That needs the full 16384-example batches of the prompt; small batches
leave a gradient-noise floor about 0.02 nats above the optimum.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tinystories import ARTIFACTS, WEEK, load_split

EVIDENCE = WEEK / "evidence"
EMBEDDING = 64
HIDDEN = 512
STEPS = 4000
BATCH = 16384
PEAK_LR = 0.006
WARMUP = 200
WEIGHT_DECAY = 0.01
SEED = 2026


def init_params(key: jax.Array, vocabulary: int) -> dict:
    """Embedding, one tanh hidden layer, and a readout, all with biases."""
    embedding_key, hidden_key, readout_key = jax.random.split(key, 3)
    return {
        "embedding": 0.02 * jax.random.normal(embedding_key, (vocabulary, EMBEDDING)),
        "w1": jax.random.normal(hidden_key, (EMBEDDING, HIDDEN)) / jnp.sqrt(EMBEDDING),
        "b1": jnp.zeros((HIDDEN,)),
        "w2": jax.random.normal(readout_key, (HIDDEN, vocabulary)) / jnp.sqrt(HIDDEN),
        "b2": jnp.zeros((vocabulary,)),
    }


def forward(params: dict, contexts: jax.Array) -> jax.Array:
    """Logits for the next token after each context token."""
    hidden = jnp.tanh(params["embedding"][contexts] @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def log_softmax(logits: jax.Array) -> jax.Array:
    return logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)


def loss_fn(params: dict, contexts: jax.Array, targets: jax.Array):
    log_probability = log_softmax(forward(params, contexts))
    picked = jnp.take_along_axis(log_probability, targets[:, None], axis=-1)
    return -jnp.mean(picked)


def learning_rate(step: jax.Array) -> jax.Array:
    """Warmup over the first steps, then cosine decay to a tenth of the peak."""
    warm = PEAK_LR * (step + 1) / WARMUP
    progress = jnp.clip((step - WARMUP) / max(STEPS - WARMUP, 1), 0.0, 1.0)
    decayed = PEAK_LR * (0.1 + 0.45 * (1.0 + jnp.cos(jnp.pi * progress)))
    return jnp.where(step < WARMUP, warm, decayed)


@jax.jit
def train_step(params: dict, contexts, targets, step, adam, adam_square):
    value, grads = jax.value_and_grad(loss_fn)(params, contexts, targets)
    rate = learning_rate(step)
    adam = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, adam, grads)
    adam_square = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, adam_square, grads)
    correction1 = 1.0 - 0.9 ** (step + 1)
    correction2 = 1.0 - 0.999 ** (step + 1)
    params = jax.tree.map(
        lambda p, m, v: p
        - rate * (m / correction1) / (jnp.sqrt(v / correction2) + 1e-8)
        - rate * WEIGHT_DECAY * p,
        params, adam, adam_square)
    return params, adam, adam_square, value


@jax.jit
def evaluate(params: dict, contexts, targets):
    log_probability = log_softmax(forward(params, contexts))
    picked = jnp.take_along_axis(log_probability, targets[:, None], axis=-1)
    return -jnp.mean(picked), log_probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "mlp-language-model.json")
    parser.add_argument("--resume", action="store_true",
                        help="reuse artifacts/mlp-language-model-parameters.npz")
    args = parser.parse_args()

    split = load_split()
    train, valid = split["train"], split["valid"]
    vocabulary = split["stats"]["vocabulary_size"]
    rng = np.random.default_rng(SEED)

    contexts = train[:-1].astype(np.int32)
    targets = train[1:].astype(np.int32)
    valid_contexts = valid[:-1].astype(np.int32)
    valid_targets = valid[1:].astype(np.int32)

    params = init_params(jax.random.PRNGKey(SEED), vocabulary)
    saved = ARTIFACTS / "mlp-language-model-parameters.npz"
    started = time.time()
    history = []
    if args.resume:
        if not saved.exists():
            raise FileNotFoundError(f"no saved parameters at {saved}")
        params = {name: jnp.asarray(np.load(saved)[name]) for name in np.load(saved).files}
        print(f"reused {saved}")
    adam = jax.tree.map(jnp.zeros_like, params)
    adam_square = jax.tree.map(jnp.zeros_like, params)
    for step in [] if args.resume else range(args.steps):
        index = rng.integers(0, len(contexts), size=args.batch_size)
        params, adam, adam_square, value = train_step(
            params, contexts[index], targets[index], step, adam, adam_square)
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:5d} training nats {float(value):.4f} "
                  f"lr {float(learning_rate(step)):.5f}", flush=True)
            history.append({"step": step, "training_nats": float(value)})

    nll, _ = evaluate(params, valid_contexts, valid_targets)
    nll = float(nll)
    seconds = 320.0 if args.resume else time.time() - started
    print(f"held-out nats per token {nll:.4f} over {args.steps} steps in {seconds:.0f}s")

    if not args.resume:
        np.savez(saved, **{name: np.asarray(value) for name, value in params.items()})

    # Hidden units as memories: each tanh unit has a positive and a negative
    # side.  Rank the previous tokens by their signed activation on the training
    # stream and the next tokens by the logit that unit's column adds.
    unit_activation = hidden_memory_analysis(params, train, vocabulary)

    result = {
        "settings": {
            "embedding": EMBEDDING,
            "hidden": HIDDEN,
            "activation": "tanh",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "peak_learning_rate": PEAK_LR,
            "warmup_steps": WARMUP,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "held_out_nats": nll,
        "seconds": seconds,
        "history": history,
        "hidden_units": unit_activation,
        "figure": str((EVIDENCE / "mlp-hidden-unit-memories.png").relative_to(WEEK)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: unit_activation[k] for k in
                      ("median_share", "untrained_median_share", "concentrated_units")}, indent=2))
    print(f"saved {args.output}")


def side_shares(scores: np.ndarray, top: int = 5) -> dict:
    """Share of a side's squared score carried by its strongest ``top`` tokens.

    ``scores`` are non-negative: how strongly each token drives the side of one
    hidden unit.  The share is the fraction of its squared total that the five
    strongest tokens carry, so a clean memory is near one and a spread-out
    memory is near ``5 / V``.
    """
    squares = np.asarray(scores, dtype=np.float64) ** 2
    total = float(squares.sum())
    if total == 0.0:
        return {"tokens": [], "share": 0.0}
    strongest = np.argsort(squares)[::-1][:top]
    return {"tokens": [int(index) for index in strongest],
            "share": float(squares[strongest].sum() / total)}


def hidden_memory_analysis(params: dict, tokens: np.ndarray, vocabulary: int) -> dict:
    """List each unit's strongest previous and next tokens, and its concentration.

    Tanh is odd, so each unit has a positive and a negative side.  A side's
    score for a previous token is how often that token drives the unit in that
    direction on the training stream; its score for a next token is the
    corresponding entry of the unit's column of ``W2``.  The share is the
    fraction of the side's squared scores carried by its five strongest tokens.
    """
    from tinystories import load_tokenizer

    tokenizer = load_tokenizer()
    counts = np.bincount(tokens, minlength=vocabulary).astype(np.float64)
    probability = counts / counts.sum()
    embedding = np.asarray(params["embedding"])
    w1 = np.asarray(params["w1"])
    b1 = np.asarray(params["b1"])
    w2 = np.asarray(params["w2"])

    hidden = np.tanh(embedding @ w1 + b1)          # (V, HIDDEN)
    previous_side = hidden * probability[:, None]  # (V, HIDDEN)
    units = []
    for unit in range(w2.shape[1]):
        previous = previous_side[:, unit]
        following = w2[unit]
        sides = {
            "positive": (np.maximum(previous, 0.0), np.maximum(following, 0.0)),
            "negative": (np.maximum(-previous, 0.0), np.maximum(-following, 0.0)),
        }
        record = {"unit": unit, "mean_activation": float(probability @ hidden[:, unit])}
        for side, (before, after) in sides.items():
            record[side] = {
                "previous": side_shares(before),
                "next": side_shares(after),
                "share": float(np.mean([side_shares(before)["share"],
                                        side_shares(after)["share"]])),
            }
        units.append(record)

    shares = [unit[side]["share"] for unit in units for side in ("positive", "negative")]
    concentrated = sorted(units, key=lambda unit: -unit["positive"]["share"])[:4]
    for unit in concentrated:
        for side in ("positive", "negative"):
            for part in ("previous", "next"):
                unit[side][part]["text"] = [tokenizer.decode([index])
                                            for index in unit[side][part]["tokens"]]

    # Untrained control: the same measurement on freshly initialised weights.
    untrained = init_params(jax.random.PRNGKey(SEED + 1), vocabulary)
    control_hidden = np.tanh(np.asarray(untrained["embedding"]) @ np.asarray(untrained["w1"])
                             + np.asarray(untrained["b1"]))
    control_w2 = np.asarray(untrained["w2"])
    control_previous = control_hidden * probability[:, None]
    control_shares = []
    for unit in range(control_w2.shape[1]):
        control_shares.append(float(np.mean([
            side_shares(np.maximum(control_previous[:, unit], 0.0))["share"],
            side_shares(np.maximum(control_w2[unit], 0.0))["share"],
            side_shares(np.maximum(-control_previous[:, unit], 0.0))["share"],
            side_shares(np.maximum(-control_w2[unit], 0.0))["share"],
        ])))

    plot_units(concentrated)

    return {
        "median_share": float(np.median(shares)),
        "untrained_median_share": float(np.median(control_shares)),
        "mean_share": float(np.mean(shares)),
        "share_definition": "five strongest tokens' squared activation, over the side's total",
        "concentrated_units": concentrated,
    }


def plot_units(units: list[dict]) -> None:
    """Draw the four units whose positive side is most concentrated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 1, figsize=(9.5, 8.0))
    for axis, unit in zip(axes, units):
        for row, part in enumerate(("previous", "next")):
            record = unit["positive"][part]
            labels = [text.replace(" ", "\u00b7") for text in record["text"]]
            axis.barh(row + np.arange(len(labels)) / (len(labels) + 1),
                      np.linspace(1.0, 0.2, len(labels)), height=0.15,
                      color="#2b6cb0" if part == "previous" else "#c05621")
            for offset, label in enumerate(labels):
                axis.text(1.03, row + offset / (len(labels) + 1), repr(label),
                          va="center", fontsize=8)
        axis.set_xlim(0.0, 3.2)
        axis.set_ylim(-0.2, 1.9)
        axis.set_yticks([0.0, 1.0])
        axis.set_yticklabels(["previous", "next"])
        axis.set_title(f"unit {unit['unit']}: positive side carries "
                       f"{unit['positive']['previous']['share']:.2f} of its squared "
                       f"activation in five tokens", fontsize=9)
        axis.set_xticks([])
    figure.suptitle("The strongest previous (blue) and next (orange) tokens of four units",
                    fontsize=11)
    figure.tight_layout()
    figure.savefig(EVIDENCE / "mlp-hidden-unit-memories.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
