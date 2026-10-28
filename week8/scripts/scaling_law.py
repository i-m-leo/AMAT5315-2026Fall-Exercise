#!/usr/bin/env python3
"""Part 6: refit the Chinchilla scaling law and derive the compute-optimal split.

The law is ``L(N, D) = E + A / N^alpha + B / D^beta`` with ``D = C / (6N)``.
It is fitted by the Huber loss of ``log L`` (delta = 0.001) from L-BFGS started
on a grid of values; 400 bootstrap resamples give the standard errors.  The
compute-optimal split comes from minimising the fitted law at fixed
``C = 6 N D``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from tinystories import ARTIFACTS, DATA, WEEK

EVIDENCE = WEEK / "evidence"
RUNS = DATA / "chinchilla-runs.csv"
DROPPED = 5
HUBER_DELTA = 0.001
BOOTSTRAP = 400
BOOTSTRAP_SEED = 2026
BUDGET = 5.76e23
PRINTED = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}


def load_runs(path: Path = RUNS) -> dict:
    """Read N, C, L, drop the five highest-loss runs, and compute D = C/(6N)."""
    import csv

    model_size, compute, loss = [], [], []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            model_size.append(float(row["Model Size"]))
            compute.append(float(row["Training FLOP"]))
            loss.append(float(row["loss"]))
    n = np.asarray(model_size)
    c = np.asarray(compute)
    l = np.asarray(loss)
    keep = np.argsort(l)[:-DROPPED]
    return {
        "all": {"N": n, "C": c, "L": l},
        "N": n[keep],
        "C": c[keep],
        "L": l[keep],
        "D": c[keep] / (6.0 * n[keep]),
        "dropped": {"N": n[np.argsort(l)[-DROPPED:]], "C": c[np.argsort(l)[-DROPPED:]],
                    "L": l[np.argsort(l)[-DROPPED:]]},
    }


def predict(parameters: dict, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    """The scaling law at model size ``n`` and token count ``d``."""
    return (parameters["E"] + parameters["A"] / n ** parameters["alpha"]
            + parameters["B"] / d ** parameters["beta"])


def pack(parameters: dict) -> np.ndarray:
    return np.log(np.asarray([parameters["E"], parameters["A"], parameters["B"],
                              parameters["alpha"], parameters["beta"]]))


def unpack(vector: np.ndarray) -> dict:
    values = np.exp(vector)
    return {"E": values[0], "A": values[1], "B": values[2],
            "alpha": values[3], "beta": values[4]}


def residuals(vector: np.ndarray, n: np.ndarray, d: np.ndarray,
              measured: np.ndarray) -> np.ndarray:
    """``log L`` residuals: predicted minus measured."""
    return np.log(predict(unpack(vector), n, d)) - np.log(measured)


def huber_fit(n, d, measured, start: dict) -> dict:
    """Minimise the Huber loss of the log-loss residual from one start."""
    result = optimize.minimize(
        lambda vector: float(np.sum(_huber(residuals(vector, n, d, measured)))),
        pack(start), method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-14, "maxiter": 20_000},
        bounds=[(np.log(0.1), np.log(10.0)), (np.log(1e-3), np.log(1e6)),
                (np.log(1e-3), np.log(1e6)), (np.log(0.01), np.log(1.5)),
                (np.log(0.01), np.log(1.5))])
    return unpack(result.x)


def _huber(r: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    """Huber loss: ``r^2/2`` inside ``delta``, linear beyond it."""
    absolute = np.abs(r)
    return np.where(absolute <= delta, 0.5 * r ** 2, delta * (absolute - 0.5 * delta))


def grid_fit(n, d, measured) -> tuple[dict, list[dict]]:
    """L-BFGS from a grid of starts; keep the result with the lowest Huber loss."""
    starts = []
    for e in (1.0, 1.7, 2.5):
        for a in (100.0, 400.0, 1000.0):
            for b in (400.0, 2000.0, 8000.0):
                for alpha in (0.2, 0.35, 0.5):
                    for beta in (0.2, 0.35, 0.5):
                        starts.append((e, a, b, alpha, beta))
    best, best_loss, tried = None, np.inf, []
    for e, a, b, alpha, beta in starts:
        start = {"E": e, "A": a, "B": b, "alpha": alpha, "beta": beta}
        fitted = huber_fit(n, d, measured, start)
        value = float(np.sum(_huber(residuals(pack(fitted), n, d, measured))))
        tried.append({"start": start, "fit": fitted, "huber": value})
        if value < best_loss:
            best, best_loss = fitted, value
    return best, tried


def bootstrap(n, d, measured, best: dict, draws: int = BOOTSTRAP,
              seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Refit on resamples of the runs, starting from the best fit."""
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(draws):
        index = rng.integers(0, len(n), size=len(n))
        fits.append(huber_fit(n[index], d[index], measured[index], best))
    keys = ("E", "A", "B", "alpha", "beta")
    return np.asarray([[fit[key] for key in keys] for fit in fits])


def optimal_split(parameters: dict, compute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``N*`` and ``D*`` that minimise the fitted law at ``C = 6 N D``."""
    alpha, beta = parameters["alpha"], parameters["beta"]
    a, b = parameters["A"], parameters["B"]
    g = (alpha * a / (beta * b)) ** (1.0 / (alpha + beta))
    budget = compute / 6.0
    n_star = g * budget ** (beta / (alpha + beta))
    d_star = budget / (g * n_star) * n_star ** 0  # D* = (1/G) (C/6)^(alpha/(alpha+beta))
    d_star = (1.0 / g) * budget ** (alpha / (alpha + beta))
    return n_star, d_star


def main() -> None:
    runs = load_runs()
    n, d, measured = runs["N"], runs["D"], runs["L"]
    print(f"{len(n)} runs fitted, {len(runs['dropped']['L'])} dropped")

    best, tried = grid_fit(n, d, measured)
    print(f"best fit: E={best['E']:.4f} A={best['A']:.1f} B={best['B']:.1f} "
          f"alpha={best['alpha']:.4f} beta={best['beta']:.4f}")

    draws = bootstrap(n, d, measured, best)
    names = ("E", "A", "B", "alpha", "beta")
    summary = {}
    for index, name in enumerate(names):
        summary[name] = {"value": best[name], "standard_error": float(draws[:, index].std())}
        print(f"  {name:6s} {best[name]:10.4f} +- {summary[name]['standard_error']:.4f}")

    compute_grid = np.logspace(18, 25, 60)
    n_star, d_star = optimal_split(best, compute_grid)
    n_printed, d_printed = optimal_split(PRINTED, compute_grid)
    n_boot, d_boot = optimal_split(best, np.asarray([BUDGET]))
    ratio = float(d_boot[0] / n_boot[0])
    ratios = []
    for row in draws:
        ns, ds = optimal_split({name: row[index] for index, name in enumerate(names)},
                               np.asarray([BUDGET]))
        ratios.append(ds[0] / ns[0])
    ratios = np.asarray(ratios)
    interval = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))
    slope_n = float(np.polyfit(np.log(compute_grid), np.log(n_star), 1)[0])
    slope_d = float(np.polyfit(np.log(compute_grid), np.log(d_star), 1)[0])
    printed_ratio = float(optimal_split(PRINTED, np.asarray([BUDGET]))[1][0]
                          / optimal_split(PRINTED, np.asarray([BUDGET]))[0][0])
    print(f"at C={BUDGET:.3g}: D*/N* = {ratio:.1f}, 95% interval {interval}, "
          f"printed law {printed_ratio:.1f}")

    fit_residual = float(np.mean(residuals(pack(best), n, d, measured)))
    printed_residual = float(np.mean(residuals(pack(PRINTED), n, d, measured)))
    print(f"mean log residual: fit {fit_residual:+.4f}, printed {printed_residual:+.4f}")

    plot(runs, best, n_star, d_star, n_printed, d_printed)

    result = {
        "runs": len(n),
        "dropped": DROPPED,
        "huber_delta": HUBER_DELTA,
        "best_fit": best,
        "standard_errors": summary,
        "bootstrap_draws": BOOTSTRAP,
        "grid": {"starts": len(tried),
                 "huber": min(entry["huber"] for entry in tried)},
        "mean_log_residual_fit": fit_residual,
        "mean_log_residual_printed": printed_residual,
        "printed_parameters": PRINTED,
        "compute_optimal": {
            "budget_flop": BUDGET,
            "model_size": float(n_boot[0]),
            "tokens": float(d_boot[0]),
            "tokens_per_parameter": ratio,
            "tokens_per_parameter_interval": interval,
            "printed_tokens_per_parameter": printed_ratio,
            "model_size_slope": slope_n,
            "tokens_slope": slope_d,
        },
        "figure": str((EVIDENCE / "chinchilla-scaling-law.png").relative_to(WEEK)),
    }
    output = ARTIFACTS / "scaling-law.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {output}")


def plot(runs: dict, best: dict, n_star, d_star, n_printed, d_printed) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    n, d, measured = runs["N"], runs["D"], runs["L"]
    compute = 6.0 * n * d

    axis = axes[0]
    colours = {1e19: "#2b6cb0", 1e20: "#2f855a", 1e21: "#c05621"}
    for budget, colour in colours.items():
        near = np.abs(np.log10(compute) - np.log10(budget)) < np.log10(1.15)
        axis.scatter(n[near], measured[near], s=18, color=colour, alpha=0.8,
                     label=f"C ~ {budget:.0e}")
        sizes = np.geomspace(n[near].min(), n[near].max(), 60) if near.any() else None
        if sizes is not None:
            axis.plot(sizes, predict(best, sizes, budget / (6.0 * sizes)),
                      color=colour, lw=1.2)
    axis.set_xscale("log")
    axis.set_xlabel("model size N")
    axis.set_ylabel("final loss L")
    axis.set_title("Chinchilla runs with the fitted law at fixed compute")
    axis.grid(alpha=0.3, which="both")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    fitted = predict(best, n, d)
    axis.scatter(fitted, measured, s=14, c=np.log10(compute), cmap="viridis")
    limit = [measured.min() * 0.97, measured.max() * 1.03]
    axis.plot(limit, limit, color="#4a5568", ls="--", lw=1)
    axis.set_xlabel("fitted loss")
    axis.set_ylabel("measured loss")
    axis.set_title("Measured against fitted loss")
    axis.grid(alpha=0.3)

    axis = axes[2]
    grid = np.logspace(18, 25, 200)
    ns, ds = optimal_split(best, grid)
    nsp, dsp = optimal_split(PRINTED, grid)
    axis.plot(ns, ds, color="#2b6cb0", lw=1.6, label=f"refit: {best['alpha']*0+1:.0f}")
    axis.plot(nsp, dsp, color="#c05621", lw=1.2, ls="--")
    axis.scatter([70e9], [1.4e12], marker="*", s=180, color="#805ad5", zorder=5,
                 label="Chinchilla (70B, 1.4T)")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("compute-optimal model size N*")
    axis.set_ylabel("compute-optimal tokens D*")
    axis.set_title(f"Compute-optimal split (D*/N* = {d_star[-1] / n_star[-1]:.0f} at the top)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(frameon=False, fontsize=8, loc="upper left")

    figure.tight_layout()
    figure.savefig(EVIDENCE / "chinchilla-scaling-law.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
