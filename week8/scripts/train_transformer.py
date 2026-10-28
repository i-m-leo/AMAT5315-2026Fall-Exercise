#!/usr/bin/env python3
"""Part 5: train the two-layer transformer on the BPE TinyStories training split.

Two layers, width 96, four heads, context 64; 6000 AdamW steps of 32 sequences
of 65 consecutive tokens, learning rate 0.003 with 300 warmup steps and cosine
decay to a tenth.  Causal attention lets one forward pass train a prediction at
every position, so each step trains 32 x 64 predictions.
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
from transformer import (CONTEXT, HEADS, LAYERS, WIDTH, forward, init_params,
                         make_train_step, position_losses, window_batch)

EVIDENCE = WEEK / "evidence"
STEPS = 6000
BATCH = 32
PEAK_LR = 0.003
WARMUP = 300
SEED = 2026


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "transformer")
    parser.add_argument("--resume", action="store_true",
                        help="evaluate the parameters already saved in --output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    split = load_split()
    train, valid = split["train"], split["valid"]
    vocabulary = split["stats"]["vocabulary_size"]
    characters_per_token = split["stats"]["valid"]["characters_per_token"]
    rng = np.random.default_rng(SEED)

    params = init_params(jax.random.PRNGKey(SEED), vocabulary)
    saved = args.output / "parameters.npz"
    started = time.time()
    history = []
    if args.resume:
        if not saved.exists():
            raise FileNotFoundError(f"no saved parameters at {saved}")
        params = unflatten(np.load(saved))
        print(f"resumed parameters from {saved}")
    adam = jax.tree.map(jnp.zeros_like, params)
    adam_square = jax.tree.map(jnp.zeros_like, params)
    train_step = make_train_step(PEAK_LR, WARMUP, args.steps)
    for step in [] if args.resume else range(args.steps):
        batch = jnp.asarray(window_batch(train, BATCH, CONTEXT, rng))
        params, adam, adam_square, value = train_step(
            params, batch, step, adam, adam_square)
        if step % 250 == 0 or step == args.steps - 1:
            history.append({"step": step, "training_nats": float(value),
                            "seconds": time.time() - started})
            print(f"step {step:5d} training nats {float(value):.4f} "
                  f"({time.time() - started:.0f}s)", flush=True)

    if not args.resume:
        np.savez(saved, **{name: np.asarray(value) for name, value in flatten(params).items()})
        print(f"saved {saved}", flush=True)

    # Held-out loss over every position of consecutive 64-token windows.
    usable = (len(valid) - 1) // CONTEXT * CONTEXT
    windows = np.concatenate((valid[:usable].reshape(-1, CONTEXT),
                              valid[1:usable + 1].reshape(-1, CONTEXT)[:, -1:]), axis=1)
    total, count = 0.0, 0
    for start in range(0, len(windows), 256):
        logits = forward(params, jnp.asarray(windows[start:start + 256, :-1]))
        log_probability = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        picked = jnp.take_along_axis(
            log_probability, jnp.asarray(windows[start:start + 256, 1:])[..., None], axis=-1)
        total += float(-picked.sum())
        count += picked.size
    held_out_nats = total / count
    held_out_bpc = held_out_nats / np.log(2.0) / characters_per_token
    seconds = time.time() - started
    print(f"held-out {held_out_nats:.4f} nats per token, {held_out_bpc:.4f} bits per character "
          f"({seconds:.0f}s)")

    per_position, standard_error = position_losses(params, valid)
    result = {
        "architecture": {"layers": LAYERS, "width": WIDTH, "heads": HEADS,
                         "context": CONTEXT, "mlp_width": 4 * WIDTH,
                         "activation": "gelu", "norm": "pre-norm", "positions": "learned"},
        "training": {"steps": args.steps, "batch_sequences": BATCH,
                     "tokens_per_step": BATCH * CONTEXT, "peak_learning_rate": PEAK_LR,
                     "warmup_steps": WARMUP, "decay": "cosine to 0.1", "seed": SEED},
        "held_out_nats": held_out_nats,
        "held_out_bits_per_character": held_out_bpc,
        "position_losses": per_position.tolist(),
        "position_standard_errors": standard_error.tolist(),
        "history": history,
        "seconds": seconds,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {args.output}/metrics.json and parameters.npz")


def unflatten(data) -> dict:
    """Rebuild the parameter tree from the flat arrays saved by :func:`flatten`."""
    names = data.files if hasattr(data, "files") else data.keys()
    flat = {name: jnp.asarray(data[name]) for name in names}
    root = {}
    for name, value in flat.items():
        parts = name.split("/")
        node = root
        for part in parts[:-1]:
            key = int(part) if part.isdigit() else part
            node = node.setdefault(key, {})
        tail = parts[-1]
        node[int(tail) if tail.isdigit() else tail] = value
    return _to_structure(root, LAYERS)


def _to_structure(node, layers: int):
    """Turn the numbered dictionaries back into lists and tuples."""
    if isinstance(node, dict):
        ordered = {key: _to_structure(value, layers) for key, value in node.items()}
        if ordered and all(isinstance(key, int) for key in ordered):
            items = [ordered[key] for key in sorted(ordered)]
            # Linear maps are (weight, bias) tuples; block lists are lists.
            return tuple(items) if len(items) == 2 else items
        return ordered
    return node


def flatten(tree: dict, prefix: str = "") -> dict:
    """Flatten the parameter tree into ``name -> array`` for npz storage."""
    flat = {}
    for name, value in tree.items():
        path = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(flatten(value, path + "/"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                flat.update(flatten(item, f"{path}/{index}/"))
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                flat[f"{path}/{index}"] = item
        else:
            flat[path] = value
    return flat


if __name__ == "__main__":
    main()
