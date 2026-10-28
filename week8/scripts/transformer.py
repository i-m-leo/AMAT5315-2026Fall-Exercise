#!/usr/bin/env python3
"""Part 5: a pre-norm transformer decoder in JAX.

Assembled from the two memories of Parts 3 and 4: a causal multi-head attention
sublayer (keys and values written by the sequence) and a GELU MLP four times
wider (memories written into the weights), each wrapped in a residual, with
layer norm before each sublayer.  Learned position vectors are added to the
token embeddings, and a readout maps the final vectors to one logit per token.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

LAYERS = 2
WIDTH = 96
HEADS = 4
CONTEXT = 64

# Shorthands for the parameter tree: each linear map is (weight, bias).
Linear = tuple[jax.Array, jax.Array]


def layer_norm(x: jax.Array, gain: jax.Array, bias: jax.Array,
               epsilon: float = 1e-5) -> jax.Array:
    """Rescale each row to mean zero and variance one, then scale and shift."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return gain * (x - mean) / jnp.sqrt(variance + epsilon) + bias


def linear(x: jax.Array, parameters: Linear) -> jax.Array:
    weight, bias = parameters
    return x @ weight + bias


def attention(x: jax.Array, block: dict, heads: int = HEADS):
    """Multi-head causal attention over the rows of ``x``.

    ``x`` has shape ``(batch, length, width)``.  Every head divides its scores
    by the square root of its own width, and masked scores are set to ``-inf``
    so their weights are exactly zero.  Returns the mixed vectors and the
    attention weights of shape ``(batch, heads, length, length)``.
    """
    batch, length, width = x.shape
    head_width = width // heads
    query = linear(x, block["wq"]).reshape(batch, length, heads, head_width)
    key = linear(x, block["wk"]).reshape(batch, length, heads, head_width)
    value = linear(x, block["wv"]).reshape(batch, length, heads, head_width)

    scores = jnp.einsum("bthd,bshd->bhts", query, key) / jnp.sqrt(head_width)
    mask = jnp.triu(jnp.full((length, length), -jnp.inf), k=1)
    weights = jax.nn.softmax(scores + mask[None, None, :, :], axis=-1)
    mixed = jnp.einsum("bhts,bshd->bthd", weights, value).reshape(batch, length, width)
    return linear(mixed, block["wo"]), weights


def gelu(u: jax.Array) -> jax.Array:
    """The tanh approximation of GELU."""
    return 0.5 * u * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (u + 0.044715 * u ** 3)))


def init_params(key: jax.Array, vocabulary: int, layers: int = LAYERS,
                width: int = WIDTH, context: int = CONTEXT) -> dict:
    """Initialise embeddings, positions, ``layers`` blocks and the readout."""
    keys = jax.random.split(key, 6 + 8 * layers)
    index = 0
    scale = lambda fan_in: 1.0 / jnp.sqrt(fan_in)

    def matrix(shape, fan_in):
        nonlocal index
        value = scale(fan_in) * jax.random.normal(keys[index], shape)
        index += 1
        return value

    params = {
        "embedding": matrix((vocabulary, width), width),
        "position": matrix((context, width), width),
        "blocks": [],
    }
    for _ in range(layers):
        params["blocks"].append({
            "norm1": (jnp.ones((width,)), jnp.zeros((width,))),
            "wq": (matrix((width, width), width), jnp.zeros((width,))),
            "wk": (matrix((width, width), width), jnp.zeros((width,))),
            "wv": (matrix((width, width), width), jnp.zeros((width,))),
            "wo": (matrix((width, width), width), jnp.zeros((width,))),
            "norm2": (jnp.ones((width,)), jnp.zeros((width,))),
            "w1": (matrix((width, 4 * width), width), jnp.zeros((4 * width,))),
            "w2": (matrix((4 * width, width), 4 * width), jnp.zeros((width,))),
        })
    params["norm_final"] = (jnp.ones((width,)), jnp.zeros((width,)))
    params["readout"] = (matrix((width, vocabulary), width), jnp.zeros((vocabulary,)))
    return params


def forward(params: dict, tokens: jax.Array, heads: int = HEADS,
            return_attention: bool = False):
    """Logits at every position, given ``tokens`` of shape (batch, length)."""
    length = tokens.shape[1]
    if length > CONTEXT:
        raise ValueError(f"window length {length} exceeds the context {CONTEXT}")
    x = params["embedding"][tokens] + params["position"][:length][None, :, :]
    attentions = []
    for block in params["blocks"]:
        normed = layer_norm(x, *block["norm1"])
        mixed, weights = attention(normed, block, heads)
        attentions.append(weights)
        x = x + mixed
        normed = layer_norm(x, *block["norm2"])
        hidden = gelu(linear(normed, block["w1"]))
        x = x + linear(hidden, block["w2"])
    x = layer_norm(x, *params["norm_final"])
    logits = linear(x, params["readout"])
    if return_attention:
        return logits, attentions
    return logits


def loss_fn(params: dict, tokens: jax.Array) -> jax.Array:
    """Mean cross-entropy over every next-token prediction of a window."""
    logits = forward(params, tokens[:, :-1])
    targets = tokens[:, 1:]
    log_probability = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    picked = jnp.take_along_axis(log_probability, targets[..., None], axis=-1)
    return -jnp.mean(picked)


def learning_rate(step, peak: float, warmup: int, total: int,
                  final_fraction: float = 0.1):
    """Linear warmup, then cosine decay to ``final_fraction`` of the peak."""
    warm = peak * (step + 1) / warmup
    progress = jnp.clip((step - warmup) / max(total - warmup, 1), 0.0, 1.0)
    low = peak * final_fraction
    decayed = low + 0.5 * (peak - low) * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < warmup, warm, decayed)


def make_train_step(peak: float, warmup: int, total: int, weight_decay: float = 0.01):
    @jax.jit
    def train_step(params, tokens, step, adam, adam_square):
        value, grads = jax.value_and_grad(loss_fn)(params, tokens)
        rate = learning_rate(step, peak, warmup, total)
        adam = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, adam, grads)
        adam_square = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, adam_square, grads)
        correction1 = 1.0 - 0.9 ** (step + 1)
        correction2 = 1.0 - 0.999 ** (step + 1)
        params = jax.tree.map(
            lambda p, m, v: p
            - rate * (m / correction1) / (jnp.sqrt(v / correction2) + 1e-8)
            - rate * weight_decay * p,
            params, adam, adam_square)
        return params, adam, adam_square, value

    return train_step


def window_batch(tokens: np.ndarray, batch_size: int, context: int,
                 rng: np.random.Generator) -> np.ndarray:
    """``batch_size`` random runs of ``context + 1`` consecutive tokens."""
    starts = rng.integers(0, len(tokens) - context - 1, size=batch_size)
    offsets = np.arange(context + 1)
    return np.asarray(tokens[starts[:, None] + offsets[None, :],], dtype=np.int32)


def position_losses(params: dict, tokens: np.ndarray, context: int = CONTEXT,
                    batch_size: int = 256, seed: int = 0) -> np.ndarray:
    """Mean loss at each of the ``context`` positions of held-out windows.

    Windows are consecutive and non-overlapping, scored from their first token,
    so position ``t`` has read ``t + 1`` tokens.
    """
    usable = (len(tokens) - 1) // context * context
    windows = tokens[:usable].reshape(-1, context)
    targets = tokens[1:usable + 1].reshape(-1, context)
    losses = np.empty((0, context))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(windows))
    for start in range(0, len(windows), batch_size):
        index = order[start:start + batch_size]
        batch = np.concatenate((windows[index], targets[index, -1:]), axis=1)
        logits = forward(params, jnp.asarray(batch[:, :-1]))
        log_probability = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        picked = jnp.take_along_axis(
            log_probability, jnp.asarray(targets[index])[..., None], axis=-1)
        losses = np.concatenate((losses, np.asarray(-picked[..., 0])), axis=0)
    return losses.mean(axis=0), losses.std(axis=0) / np.sqrt(len(losses))
