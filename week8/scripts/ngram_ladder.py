#!/usr/bin/env python3
"""Part 2: the n-gram ladder, its normalization, sampler, and curse of dimensionality.

Fits add-k smoothed n-gram models of order 1..6 on the BPE training tokens,
chooses the smoothing constant on the last 10% of those tokens, and reports the
held-out cross-entropy, perplexity and bits per character on the validation
stories.  It also checks that the bigram is exactly normalized by summing over
all sequences of length 1, 2 and 3, and that bigram ancestral samples follow the
model by a pooled chi-squared test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import ngram
from tinystories import ARTIFACTS, WEEK, load_split

EVIDENCE = WEEK / "evidence"
SAMPLER_TOKENS = 250_000
SAMPLER_TOP_CONTEXTS = 20
SAMPLER_SEED = 1848
UNIFORM_NATS = float(np.log(512.0))


def normalization_check(count_set: ngram.CountSet, smoothing: float) -> dict:
    """Sum the bigram probability over every sequence of length 1, 2 and 3."""
    vocabulary = count_set.vocabulary_size
    start = count_set.distribution(2, ngram.EOS_ID, smoothing)  # p(x1 | EOS)
    transition = np.empty((vocabulary, vocabulary))
    for context in range(vocabulary):
        transition[context] = count_set.distribution(2, context, smoothing)

    sums = {}
    distribution = start
    for length in (1, 2, 3):
        sums[str(length)] = float(distribution.sum())
        distribution = distribution @ transition

    # The deliberately wrong scores of the sheet: exp(n(a b) / max n), never
    # divided by their row sum.  The same enumeration then grows like V**length.
    table = count_set.tables[2]
    counts = np.zeros((vocabulary, vocabulary))
    rows, columns = np.divmod(table["pair_keys"], vocabulary)
    counts[rows, columns] = table["pair_counts"]
    scale = np.exp(counts / counts.max())
    start_scores = np.exp(
        np.bincount(count_set.train_tokens, minlength=vocabulary)
        / np.bincount(count_set.train_tokens, minlength=vocabulary).max())
    wrong = {}
    distribution = start_scores
    for length in (1, 2, 3):
        wrong[str(length)] = float(distribution.sum())
        distribution = distribution @ scale

    return {
        "sequence_sums": sums,
        "max_abs_error": max(abs(value - 1.0) for value in sums.values()),
        "unnormalized_sequence_sums": wrong,
    }


def chi_squared_sampler_check(count_set: ngram.CountSet, validation: np.ndarray,
                              smoothing: float, rng: np.random.Generator) -> dict:
    """Compare 250k bigram samples with the model, pooled over 20 contexts."""
    frequency = np.bincount(validation, minlength=count_set.vocabulary_size)
    top = np.argsort(frequency)[::-1][:SAMPLER_TOP_CONTEXTS]

    # One long ancestral walk; pairs (context, next) are the draws.
    drawn = ngram.sample_markov_chain(rng, count_set, 2,
                                      np.asarray([ngram.EOS_ID]), SAMPLER_TOKENS,
                                      smoothing)
    contexts = np.concatenate((np.asarray([ngram.EOS_ID]), drawn[:-1]))
    pairs = np.stack((contexts, drawn), axis=1)

    statistic = 0.0
    degrees = 0
    per_context = {}
    for token in top:
        following = pairs[pairs[:, 0] == token, 1]
        occurrences = len(following)
        if occurrences == 0:
            continue
        expected = occurrences * count_set.distribution(2, int(token), smoothing)
        observed = np.bincount(following, minlength=count_set.vocabulary_size).astype(np.float64)
        keep = expected >= 5.0
        if keep.sum() < 2:
            continue
        statistic += float(((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum())
        # Everything below the cutoff is one pooled bin.
        rest_expected = expected[~keep].sum()
        if rest_expected > 0:
            statistic += float((observed[~keep].sum() - rest_expected) ** 2 / rest_expected)
            degrees += 1
        degrees += int(keep.sum()) - 1
        per_context[int(token)] = {"draws": int(occurrences), "kept_bins": int(keep.sum())}

    p_value = float(stats.chi2.sf(statistic, degrees))

    # Negative control: a sampler that conditions on the token two places back.
    wrong_contexts = np.concatenate((np.asarray([ngram.EOS_ID, ngram.EOS_ID]), drawn[:-2]))
    wrong_pairs = np.stack((wrong_contexts, drawn), axis=1)
    wrong_statistic = 0.0
    wrong_degrees = 0
    for token in top:
        following = wrong_pairs[wrong_pairs[:, 0] == token, 1]
        occurrences = len(following)
        if occurrences == 0:
            continue
        expected = occurrences * count_set.distribution(2, int(token), smoothing)
        observed = np.bincount(following, minlength=count_set.vocabulary_size).astype(np.float64)
        keep = expected >= 5.0
        if keep.sum() < 2:
            continue
        wrong_statistic += float(((observed[keep] - expected[keep]) ** 2 / expected[keep]).sum())
        rest_expected = expected[~keep].sum()
        if rest_expected > 0:
            wrong_statistic += float((observed[~keep].sum() - rest_expected) ** 2 / rest_expected)
            wrong_degrees += 1
        wrong_degrees += int(keep.sum()) - 1
    wrong_p = float(stats.chi2.sf(wrong_statistic, wrong_degrees))

    return {
        "seed": SAMPLER_SEED,
        "draws": SAMPLER_TOKENS,
        "contexts": int(len(per_context)),
        "chi_squared": statistic,
        "degrees_of_freedom": degrees,
        "p_value": p_value,
        "negative_control": {
            "description": "sampler conditioned on the token two places back",
            "chi_squared": wrong_statistic,
            "degrees_of_freedom": wrong_degrees,
            "p_value": wrong_p,
        },
        "per_context": per_context,
    }


def plot(rows: list[dict], uniform_bpc: float) -> Path:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    orders = np.asarray([row["n"] for row in rows])
    bpc = np.asarray([row["bits_per_character"] for row in rows])
    unseen = np.asarray([row["unseen_fraction"] for row in rows]) * 100

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    left.axhline(uniform_bpc, color="grey", ls=":", label="uniform")
    left.plot(orders, bpc, "o-", color="#2b6cb0")
    best = int(orders[int(np.argmin(bpc))])
    left.axvline(best, color="#c53030", ls="--", lw=1)
    left.annotate(f"best n = {best}", (best, bpc.min()), textcoords="offset points",
                  xytext=(8, 10), color="#c53030")
    left.set_xlabel("n-gram order n")
    left.set_ylabel("held-out bits per character")
    left.set_title("Held-out loss")
    left.grid(alpha=0.3)
    left.legend(frameon=False)

    right.plot(orders, unseen, "s-", color="#805ad5")
    right.set_xlabel("n-gram order n")
    right.set_ylabel("% of held-out n-grams never in training")
    right.set_title("The curse of dimensionality")
    right.grid(alpha=0.3)
    figure.suptitle("Count models on the BPE TinyStories split", y=1.02)
    figure.tight_layout()
    path = EVIDENCE / "ngram-curse-of-dimensionality.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "ngram-ladder.json")
    args = parser.parse_args()

    split = load_split()
    train, valid, stats = split["train"], split["valid"], split["stats"]
    vocabulary = stats["vocabulary_size"]
    orders = list(range(1, 7))

    # Smoothing is chosen on the last 10% of the training tokens: fit on the
    # first 90% and score the tail, then refit on every training token.
    cut = int(len(train) * 0.9)
    selection_set = ngram.CountSet(train[:cut], vocabulary, orders=orders)
    tail = train[cut:]
    chosen = {}
    for order in orders:
        best, scores = ngram.choose_smoothing(selection_set, order, tail)
        chosen[order] = {"smoothing": best, "validation_nats": scores}
    del selection_set

    count_set = ngram.CountSet(train, vocabulary, orders=orders)
    characters_per_token = stats["valid"]["characters_per_token"]
    nll_cache = {}
    rows = []
    for order in orders:
        smoothing = chosen[order]["smoothing"]
        nll = count_set.nll(order, valid, smoothing)
        nll_cache[order] = nll
        nats = float(nll.mean())
        rows.append({
            "n": order,
            "smoothing": smoothing,
            "nats_per_token": nats,
            "perplexity": float(np.exp(nats)),
            "bits_per_character": float(nats / np.log(2.0) / characters_per_token),
            "unseen_fraction": count_set.unseen_fraction(order, valid),
            "contexts_seen": count_set.tables[order]["contexts_seen"],
            "possible_contexts": count_set.tables[order]["possible_contexts"],
        })
        print(f"n={order} k={smoothing:g} nats={nats:.4f} "
              f"ppl={np.exp(nats):.4g} bpc={rows[-1]['bits_per_character']:.4f} "
              f"unseen={100 * rows[-1]['unseen_fraction']:.2f}%")

    uniform = {
        "n": 0,
        "name": "uniform",
        "smoothing": None,
        "nats_per_token": UNIFORM_NATS,
        "perplexity": 512.0,
        "bits_per_character": UNIFORM_NATS / np.log(2.0) / characters_per_token,
        "unseen_fraction": 1.0,
    }
    print(f"uniform nats={UNIFORM_NATS:.4f} bpc={uniform['bits_per_character']:.4f}")

    normalization = normalization_check(count_set, chosen[2]["smoothing"])
    print("normalization:", normalization["sequence_sums"],
          "max error", normalization["max_abs_error"])

    rng = np.random.default_rng(SAMPLER_SEED)
    sampler = chi_squared_sampler_check(count_set, valid, chosen[2]["smoothing"], rng)
    print(f"sampler chi2={sampler['chi_squared']:.1f} dof={sampler['degrees_of_freedom']} "
          f"p={sampler['p_value']:.3f}; negative control p="
          f"{sampler['negative_control']['p_value']:.3g}")

    figure_path = plot(rows, uniform["bits_per_character"])
    best = min(rows, key=lambda row: row["nats_per_token"])

    result = {
        "split": stats,
        "orders": rows,
        "uniform": uniform,
        "smoothing_selection": {str(k): v for k, v in chosen.items()},
        "normalization": normalization,
        "sampler": sampler,
        "best_order": best["n"],
        "trigram_nats": next(row["nats_per_token"] for row in rows if row["n"] == 3),
        "figure": str(figure_path.relative_to(WEEK)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {args.output} and {figure_path}")


if __name__ == "__main__":
    main()
