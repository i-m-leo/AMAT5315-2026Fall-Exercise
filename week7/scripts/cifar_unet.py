#!/usr/bin/env python3
"""JAX inference for the pinned Diffusers google/ddpm-cifar10-32 U-Net.

The operations and checkpoint mapping follow diffusers v0.40.0's UNet2DModel,
ResnetBlock2D, Downsample2D, Upsample2D, and deprecated spatial Attention
blocks. Inputs and outputs use the checkpoint's (B, 3, 32, 32) layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from safetensors.numpy import load_file


def _check_config(config: dict[str, Any]) -> None:
    """Fail rather than silently misinterpret a different U-Net checkpoint."""
    expected = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "center_input_sample": False,
        "time_embedding_type": "positional",
        "freq_shift": 1,
        "flip_sin_to_cos": False,
        "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
        "block_out_channels": [128, 256, 256, 256],
        "layers_per_block": 2,
        "downsample_padding": 0,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
    }
    for name, wanted in expected.items():
        if config.get(name) != wanted:
            raise ValueError(f"unsupported CIFAR U-Net config {name}={config.get(name)!r}; expected {wanted!r}")


def load_cifar_unet(model_dir: str | Path) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Load checkpoint tensors into JAX layout and return them with config."""
    model_dir = Path(model_dir)
    with (model_dir / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    _check_config(config)
    raw = load_file(str(model_dir / "diffusion_pytorch_model.safetensors"))
    params: dict[str, jax.Array] = {}
    for name, value in raw.items():
        value = np.asarray(value, dtype=np.float32)
        if name.endswith(".weight") and value.ndim == 4:
            # PyTorch OIHW -> JAX HWIO convolution layout.
            value = value.transpose(2, 3, 1, 0)
        elif name.endswith(".weight") and value.ndim == 2:
            # PyTorch (out, in) -> JAX (in, out) dense layout.
            value = value.T
        params[name] = jnp.asarray(value)
    return params, config


def _linear(x: jax.Array, params: dict[str, jax.Array], name: str) -> jax.Array:
    return x @ params[f"{name}.weight"] + params[f"{name}.bias"]


def _conv(
    x: jax.Array,
    params: dict[str, jax.Array],
    name: str,
    *,
    stride: int = 1,
    padding: str | tuple[tuple[int, int], tuple[int, int]] = ((1, 1), (1, 1)),
) -> jax.Array:
    y = lax.conv_general_dilated(
        x,
        params[f"{name}.weight"],
        window_strides=(stride, stride),
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + params[f"{name}.bias"]


def _group_norm(
    x: jax.Array,
    params: dict[str, jax.Array],
    name: str,
    *,
    groups: int = 32,
    eps: float = 1e-6,
) -> jax.Array:
    batch, height, width, channels = x.shape
    if channels % groups:
        raise ValueError(f"{channels} channels cannot be divided into {groups} groups")
    grouped = x.reshape(batch, height, width, groups, channels // groups)
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = jnp.mean(jnp.square(grouped - mean), axis=(1, 2, 4), keepdims=True)
    normalized = ((grouped - mean) * lax.rsqrt(variance + eps)).reshape(x.shape)
    return normalized * params[f"{name}.weight"] + params[f"{name}.bias"]


def _resnet(
    x: jax.Array,
    time_embedding: jax.Array,
    params: dict[str, jax.Array],
    name: str,
) -> jax.Array:
    residual = x
    hidden = _group_norm(x, params, f"{name}.norm1")
    hidden = jax.nn.silu(hidden)
    hidden = _conv(hidden, params, f"{name}.conv1")
    time = jax.nn.silu(time_embedding)
    time = _linear(time, params, f"{name}.time_emb_proj")
    hidden = hidden + time[:, None, None, :]
    hidden = _group_norm(hidden, params, f"{name}.norm2")
    hidden = jax.nn.silu(hidden)
    hidden = _conv(hidden, params, f"{name}.conv2")
    if residual.shape[-1] != hidden.shape[-1]:
        residual = _conv(residual, params, f"{name}.conv_shortcut", padding="VALID")
    return residual + hidden


def _attention(x: jax.Array, params: dict[str, jax.Array], name: str) -> jax.Array:
    """One-head spatial self-attention matching Diffusers' deprecated block."""
    residual = x
    batch, height, width, channels = x.shape
    hidden = _group_norm(x, params, f"{name}.group_norm")
    hidden = hidden.reshape(batch, height * width, channels)
    query = _linear(hidden, params, f"{name}.query")
    key = _linear(hidden, params, f"{name}.key")
    value = _linear(hidden, params, f"{name}.value")
    scores = jnp.einsum("bqc,bkc->bqk", query, key) * (channels ** -0.5)
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
    attended = jnp.einsum("bqk,bkc->bqc", probabilities, value)
    attended = _linear(attended, params, f"{name}.proj_attn")
    return residual + attended.reshape(batch, height, width, channels)


def _downsample(x: jax.Array, params: dict[str, jax.Array], name: str) -> jax.Array:
    # Diffusers Downsample2D(padding=0) pads only the right and bottom edges.
    x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    return _conv(x, params, name, stride=2, padding="VALID")


def _upsample(x: jax.Array, params: dict[str, jax.Array], name: str) -> jax.Array:
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return _conv(x, params, name)


def cifar_unet(
    params: dict[str, jax.Array],
    sample: jax.Array,
    model_indices: jax.Array,
) -> jax.Array:
    """Run the pretrained noise predictor; model indices are zero-based."""
    sample = jnp.asarray(sample, dtype=jnp.float32)
    if sample.ndim != 4 or sample.shape[1:] != (3, 32, 32):
        raise ValueError(f"sample must have shape (batch, 3, 32, 32), got {sample.shape}")
    model_indices = jnp.broadcast_to(jnp.asarray(model_indices).reshape(-1), (sample.shape[0],))
    # Diffusers Timesteps(128, flip_sin_to_cos=False, downscale_freq_shift=1).
    half = 64
    frequencies = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
    phases = model_indices.astype(jnp.float32)[:, None] * frequencies[None, :]
    time = jnp.concatenate((jnp.sin(phases), jnp.cos(phases)), axis=-1)
    time = _linear(time, params, "time_embedding.linear_1")
    time = jax.nn.silu(time)
    time = _linear(time, params, "time_embedding.linear_2")

    hidden = jnp.transpose(sample, (0, 2, 3, 1))
    hidden = _conv(hidden, params, "conv_in")
    skips = [hidden]
    down_channels = (128, 256, 256, 256)
    down_attention = (False, True, False, False)
    for block, _channels in enumerate(down_channels):
        for layer in range(2):
            hidden = _resnet(hidden, time, params, f"down_blocks.{block}.resnets.{layer}")
            if down_attention[block]:
                hidden = _attention(hidden, params, f"down_blocks.{block}.attentions.{layer}")
            skips.append(hidden)
        if block < 3:
            hidden = _downsample(hidden, params, f"down_blocks.{block}.downsamplers.0.conv")
            skips.append(hidden)

    hidden = _resnet(hidden, time, params, "mid_block.resnets.0")
    hidden = _attention(hidden, params, "mid_block.attentions.0")
    hidden = _resnet(hidden, time, params, "mid_block.resnets.1")

    up_channels = (256, 256, 256, 128)
    up_attention = (False, False, True, False)
    for block, _channels in enumerate(up_channels):
        for layer in range(3):
            skip = skips.pop()
            hidden = jnp.concatenate((hidden, skip), axis=-1)
            hidden = _resnet(hidden, time, params, f"up_blocks.{block}.resnets.{layer}")
            if up_attention[block]:
                hidden = _attention(hidden, params, f"up_blocks.{block}.attentions.{layer}")
        if block < 3:
            hidden = _upsample(hidden, params, f"up_blocks.{block}.upsamplers.0.conv")
    if skips:
        raise AssertionError(f"U-Net left {len(skips)} unused skip tensors")
    hidden = _group_norm(hidden, params, "conv_norm_out")
    hidden = jax.nn.silu(hidden)
    output = _conv(hidden, params, "conv_out")
    return jnp.transpose(output, (0, 3, 1, 2))


def make_parameterized_predictor():
    """Create a compiled predictor with parameters passed as dynamic arguments."""
    return jax.jit(cifar_unet)
