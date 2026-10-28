#!/usr/bin/env python3
"""Part 5 verification: samplers, context use, and surprise against entropy.

* (1) The model comparison and its samples live in ``compare_models.py``.
* (2) ``context_use`` scores held-out windows from their first token and plots
  the mean loss at each position against the bigram's and trigram's.
* (3) ``sampler_stats`` draws 100000 next tokens from one fixed broad context
  with each sampler and tests the counts against the sampler's exact target.
* (4) ``self_consistency`` compares the mean surprise of the model's own
  samples with its mean predicted entropy, as a z-score over sequences.
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
from transformer import CONTEXT, forward, position_losses

EVIDENCE = WEEK / "evidence"
DRAWS = 100_000
SAMPLER_SEED = 1848
SEQUENCES = 128
SEQUENCE_LENGTH = 128


def load_transformer(path: Path):
    from train_transformer import unflatten

    data = np.load(path / "parameters.npz")
    metrics = json.loads((path / "metrics.json").read_text())
    return unflatten(data), metrics


def logits_for(tokens: np.ndarray, params) -> np.ndarray:
    """Next-token logits from a token prefix, using at most the context window."""
    window = tokens[-CONTEXT:][None, :]
    logits = forward(params, jnp.asarray(window))
    return np.asarray(logits[0, -1])


def broad_context(params, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fixed context whose next-token distribution is the broadest."""
    best = (0.0, -np.inf, None)
    for start in range(0, 400 * CONTEXT, CONTEXT):
        window = valid[start:start + CONTEXT]
        if len(window) < CONTEXT:
            break
        logits = logits_for(window, params)
        probability = samplers.softmax(logits)
        entropy = float(-(probability * np.log(probability + 1e-12)).sum())
        if entropy > best[1]:
            best = (entropy, entropy, window)
    return best[2], best[1]


def sampler_stats(params, valid: np.ndarray, rng: np.random.Generator) -> dict:
    """Compare 100k draws of each sampler with its exact target distribution."""
    context, entropy = broad_context(params, valid)
    logits = logits_for(context, params)
    probability = samplers.softmax(logits)
    tops = np.argsort(probability)[::-1][:10]
    print(f"fixed context entropy {entropy:.3f} nats; top tokens {tops.tolist()}")

    targets = {
        "temperature 0.8": samplers.temperature_distribution(logits, 0.8),
        "top-k 20 at T=0.8": samplers.top_k_distribution(logits, 20, 0.8),
        "top-p 0.9 at T=0.8": samplers.top_p_distribution(logits, 0.9, 0.8),
        "min-p 0.1 at T=0.7": samplers.min_p_distribution(logits, 0.1, 0.7),
    }
    results = {}
    for name, target in targets.items():
        drawn = samplers.draw(rng, target, DRAWS)
        counts = np.bincount(drawn, minlength=len(target)).astype(np.float64)
        statistic, degrees, p_value = samplers.chi_squared_against(target, counts)
        results[name] = {"chi_squared": statistic, "degrees_of_freedom": degrees,
                         "p_value": p_value, "support": int((target > 0).sum())}
        print(f"{name:22s} support {int((target > 0).sum()):4d} "
              f"chi2 {statistic:7.1f} dof {degrees:4d} p {p_value:.3f}")

    # Greedy decoding is the limit of temperature 0.
    greedy = samplers.draw(rng, samplers.temperature_distribution(logits, 1e-4), 200)
    results["temperature 0.0001"] = {
        "always_most_likely": bool(np.all(greedy == int(np.argmax(logits)))),
        "argmax": int(np.argmax(logits)),
    }
    # The wrong top-p sampler drops the token that crosses the threshold.
    order = np.argsort(probability)[::-1]
    cumulative = np.cumsum(probability[order])
    cutoff = int(np.searchsorted(cumulative, 0.9))  # drops the crossing token
    wrong = np.zeros_like(probability)
    wrong[order[:cutoff]] = probability[order[:cutoff]]
    wrong /= wrong.sum()
    counts = np.bincount(samplers.draw(rng, wrong, DRAWS), minlength=len(wrong)).astype(np.float64)
    results["dropping top-p (negative control)"] = dict(zip(
        ("chi_squared", "degrees_of_freedom", "p_value"),
        samplers.chi_squared_against(targets["top-p 0.9 at T=0.8"], counts)))
    print(f"dropping top-p: p={results['dropping top-p (negative control)']['p_value']:.3g}")

    results["context"] = {"tokens": context.tolist(), "entropy_nats": entropy,
                          "draws": DRAWS, "seed": SAMPLER_SEED}
    return results


def sample_story(params, start: np.ndarray, num_tokens: int,
                 rng: np.random.Generator, sampler) -> np.ndarray:
    tokens = list(np.asarray(start, dtype=np.int64))
    for _ in range(num_tokens):
        logits = logits_for(np.asarray(tokens), params)
        probability = sampler(logits)
        tokens.append(int(samplers.draw(rng, probability, 1)[0]))
    return np.asarray(tokens, dtype=np.int64)


def self_consistency(params, valid: np.ndarray, rng: np.random.Generator) -> dict:
    """Mean surprise of the model's own samples against its mean predicted entropy.

    The surprise of a drawn token is ``-ln p(v)``; averaged under the model's own
    distribution it equals the entropy ``H = -sum p ln p``.  Text sampled at a
    lower temperature is less surprising than the temperature-1 model predicts.
    """
    results = {}
    for temperature in (1.0, 0.7):
        sampler = (lambda logits: samplers.softmax(logits)) if temperature == 1.0 else \
            (lambda logits: samplers.temperature_distribution(logits, temperature))
        scores = []
        for sequence_index in range(SEQUENCES):
            start = valid[CONTEXT * sequence_index:CONTEXT * sequence_index + CONTEXT]
            generated = sample_story(params, start, SEQUENCE_LENGTH, rng, sampler)
            surprise, entropy = 0.0, 0.0
            count = 0
            for position in range(CONTEXT, len(generated)):
                logits = logits_for(generated[:position], params)
                probability = samplers.softmax(logits)  # scored by the T=1 model
                token = int(generated[position])
                surprise += -np.log(probability[token] + 1e-12)
                entropy += float(-(probability * np.log(probability + 1e-12)).sum())
                count += 1
            scores.append((surprise - entropy) / count)  # per-token difference
        difference = np.asarray(scores)
        z_score = float(difference.mean() / (difference.std(ddof=1) / np.sqrt(len(difference))))
        results[str(temperature)] = {
            "mean_surprise": float(difference.mean() + np.mean([0.0])),
            "mean_difference_per_token": float(difference.mean()),
            "standard_error": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            "z_score": z_score,
            "sequences": SEQUENCES,
        }
        print(f"T={temperature}: surprise - entropy = {difference.mean():+.5f} "
              f"+- {difference.std(ddof=1) / np.sqrt(len(difference)):.5f} per token, "
              f"z {z_score:+.1f}")
    return results


def context_use(params, model_metrics: dict, valid: np.ndarray,
                train: np.ndarray) -> dict:
    """Mean loss at each position of held-out windows, with n-gram baselines."""
    per_position, standard_error = position_losses(params, valid)
    count_set_data = json.loads((ARTIFACTS / "ngram-ladder.json").read_text())
    cm = ngram.CountSet(train, 512)
    uniform = float(np.log(512.0))
    rows = []
    for order in (1, 2, 3):
        smoothing = count_set_data["orders"][order - 1]["smoothing"]
        total = float(cm.nll(order, valid, smoothing).mean())
        rows.append({"n": order, "nats_per_token": total})
    return {
        "per_position": per_position.tolist(),
        "standard_error": standard_error.tolist(),
        "bigram": rows[1]["nats_per_token"],
        "trigram": rows[2]["nats_per_token"],
        "unigram": rows[0]["nats_per_token"],
        "uniform": uniform,
        "last8_mean": float(np.mean(per_position[-8:])),
        "held_out_nats": model_metrics["held_out_nats"],
    }


def plot_context(result: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    positions = np.arange(1, len(result["per_position"]) + 1)
    loss = np.asarray(result["per_position"])
    error = np.asarray(result["standard_error"])
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.fill_between(positions, loss - 2 * error, loss + 2 * error,
                      color="#2b6cb0", alpha=0.2, lw=0)
    axis.plot(positions, loss, color="#2b6cb0", lw=1.6, label="transformer")
    axis.axhline(result["bigram"], color="#c05621", ls="--", lw=1.0, label="bigram")
    axis.axhline(result["trigram"], color="#2f855a", ls=":", lw=1.2, label="trigram")
    axis.set_xlabel("tokens read (position in the 64-token window)")
    axis.set_ylabel("held-out nats per token")
    axis.set_title("The transformer keeps improving as it reads its context")
    axis.grid(alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = EVIDENCE / "transformer-context-use.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=ARTIFACTS / "transformer")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "transformer-checks.json")
    args = parser.parse_args()

    split = load_split()
    train, valid = split["train"], split["valid"]
    params, metrics = load_transformer(args.parameters)
    rng = np.random.default_rng(SAMPLER_SEED)

    context = context_use(params, metrics, valid, train)
    figure = plot_context(context)
    print(f"last 8 positions {context['last8_mean']:.3f} nats "
          f"(trigram {context['trigram']:.3f})")

    result = {
        "context_use": context,
        "samplers": sampler_stats(params, valid, rng),
        "self_consistency": self_consistency(params, valid, rng),
        "figure": str(figure.relative_to(WEEK)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {args.output} and {figure}")


if __name__ == "__main__":
    main()
