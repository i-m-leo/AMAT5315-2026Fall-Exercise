#!/usr/bin/env python3
"""Load the TinyStories corpus and its BPE-512 tokenizer.

The corpus is split into whole stories at ``<|endoftext|>``.  The first 90% of
the stories in file order train every model this week; the remaining 10% are
held out.  Both splits are encoded with the ``tok512`` byte-pair encoding, with
its end-of-sequence token between stories.  Cached token arrays live under
``artifacts/`` because they are large and reproducible from ``data/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WEEK = Path(__file__).resolve().parents[1]
DATA = WEEK / "data"
ARTIFACTS = WEEK / "artifacts"
CORPUS = DATA / "TinyStoriesV2-GPT4-valid.txt"
TOKENIZER = DATA / "tok512.model"
STORY_SEPARATOR = "<|endoftext|>"
TRAIN_FRACTION = 0.90


def read_stories(path: Path = CORPUS) -> list[str]:
    """Return the non-empty stories, stripped, in file order."""
    text = path.read_text(encoding="utf-8")
    stories = [piece.strip() for piece in text.split(STORY_SEPARATOR)]
    return [story for story in stories if story]


def split_stories(stories: list[str], train_fraction: float = TRAIN_FRACTION):
    """Split whole stories into the first ``train_fraction`` and the rest."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")
    boundary = int(len(stories) * train_fraction)
    return stories[:boundary], stories[boundary:]


def load_tokenizer():
    """Load the sentencepiece tok512 model (vocabulary size 512)."""
    import sentencepiece as spm

    if not TOKENIZER.exists():
        raise FileNotFoundError(
            f"missing {TOKENIZER}; download it from karpathy/tinyllamas "
            "stories260K/tok512.model at revision 0bd21da"
        )
    return spm.SentencePieceProcessor(model_file=str(TOKENIZER))


def encode_stories(tokenizer, stories: list[str]) -> np.ndarray:
    """Encode stories as int32 ids, separating consecutive stories by EOS."""
    eos = tokenizer.eos_id()
    pieces: list[np.ndarray] = []
    for story in stories:
        pieces.append(np.asarray(tokenizer.encode(story), dtype=np.int32))
        pieces.append(np.asarray([eos], dtype=np.int32))
    return np.concatenate(pieces)


def token_char_counts(tokenizer, stories: list[str]) -> dict:
    """Count decoded tokens and characters for the bits-per-character unit.

    ``decode_coverage`` counts stories whose decode (with the merged double
    spaces the BPE cannot split) reproduces the original text exactly.
    """
    stories_characters = sum(len(story) for story in stories)
    token_count = 0
    decoded_characters = 0
    exact = 0
    for story in stories:
        ids = tokenizer.encode(story)
        token_count += len(ids) + 1  # the separating EOS token
        decoded = tokenizer.decode(ids)
        decoded_characters += len(decoded) + 1
        exact += int(decoded == story)
    return {
        "stories": len(stories),
        "characters": stories_characters,
        "tokens": token_count,
        "characters_per_token": stories_characters / token_count,
        "decoded_characters": decoded_characters,
        "decode_coverage": exact / len(stories),
        "exact_decode_stories": exact,
    }


def load_split(cache: bool = True) -> dict:
    """Return the encoded splits, their statistics, and the tokenizer."""
    cache_path = ARTIFACTS / "tinystories-split.npz"
    stats_path = ARTIFACTS / "tinystories-split.json"
    if cache and cache_path.exists() and stats_path.exists():
        data = np.load(cache_path)
        stats = json.loads(stats_path.read_text())
        return {
            "train": data["train"],
            "valid": data["valid"],
            "stats": stats,
            "cached": True,
        }

    tokenizer = load_tokenizer()
    stories = read_stories()
    train_stories, valid_stories = split_stories(stories)
    train = encode_stories(tokenizer, train_stories)
    valid = encode_stories(tokenizer, valid_stories)
    stats = {
        "corpus": str(CORPUS.relative_to(WEEK)),
        "tokenizer": str(TOKENIZER.relative_to(WEEK)),
        "vocabulary_size": tokenizer.get_piece_size(),
        "eos_id": tokenizer.eos_id(),
        "total_stories": len(stories),
        "train": token_char_counts(tokenizer, train_stories),
        "valid": token_char_counts(tokenizer, valid_stories),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, train=train, valid=valid)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    return {"train": train, "valid": valid, "stats": stats, "cached": False}


if __name__ == "__main__":
    split = load_split()
    print(json.dumps(split["stats"], indent=2))
