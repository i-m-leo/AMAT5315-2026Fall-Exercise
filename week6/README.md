# Week 6

Reusable dataset visualization tools for AMAT5315-2026Fall.

## Setup

Install dependencies with `uv` (CPU-only; no ML framework or accelerator is used):

```sh
uv sync --extra test
uv run pytest
```

## Visualization

`scripts/visualize.py` reads CIFAR-10 binary archives and MNIST IDX gzip files.
It produces native-resolution grayscale/RGB image galleries and split-wise
class-frequency plots. With `--split both`, it also writes one combined
empirical-distribution figure per dataset: three deterministically sampled,
labeled training examples for every class above a train/test relative-frequency
chart. Relative frequency is the class count divided by that split's sample
count. Image arrays retain their original dimensions and integer pixel values;
plots use nearest-neighbor rendering without resize or normalization. Labels
can be hidden from the standalone galleries with `--no-labels`.

From `week6/`, for both CIFAR-10 splits (outputs saved in `evidence/`):

```sh
uv run python scripts/visualize.py cifar10 \\
  --input inputs/cifar10/cifar-10-binary.tar.gz --split both --count 20
```

For MNIST:

```sh
uv run python scripts/visualize.py mnist --input inputs/mnist --split both --count 20
```

Use `--split train` or `--split test`, `--no-labels`, or `--output-dir PATH`
to customize output; `--examples-per-class N` sets the number of examples in
the combined figure. The default output directory is `evidence/`. The reusable
functions `load_cifar10`, `load_mnist`, `image_gallery`, and `class_frequency`
are importable from `scripts/visualize.py`.

Dataset downloads and generated raw runs are ignored by git; generated plots
in `evidence/` are committed as visual evidence.

## JAX restricted Boltzmann machine

`scripts/rbm.py` implements an MNIST binary RBM with 784 visible and 128 hidden
units. Pixels are binarized at intensity 128 (`uint8 >= 128`). The joint energy
is `E(v,h) = -vb·v - hb·h - vᵀWh`, and the visible free energy is
`F(v) = -vb·v - Σ softplus(hb + vW)`. A block Gibbs transition samples hidden
units from `sigmoid(vW + hb)` and then visible units from `sigmoid(hWᵀ + vb)`.

The differentiable CD-k surrogate is the minibatch mean of
`F(v_data) - F(stop_gradient(v_k))`, where `v_k` is sampled by k binary block
Gibbs transitions. Gradients differentiate both free-energy terms with respect
to the parameters while treating the sampled negative configuration as fixed.
CD-1 and CD-10 share parameter initialization, per-epoch data permutations,
and evaluation random keys. Evaluation uses the first configured held-out
images; the one-step reconstruction error is the fraction of pixels differing
from the binarized input after one visible-hidden-visible transition.

Run the configuration in `rbm.toml` on CPU:

```sh
JAX_PLATFORMS=cpu uv run python scripts/rbm.py \\
  --config rbm.toml --data inputs/mnist --out artifacts/rbm
```

The run writes a copy of its settings and `metrics.json` under ignored
`artifacts/rbm/`. With the current config and local MNIST data, the final
(epoch 20) held-out one-step changed-pixel fractions were 0.04616 for CD-1 and
0.04851 for CD-10. Final epoch-mean batch CD surrogates were -5.83286 and
-4.92241, respectively. `uv run pytest` covers the energy/free-energy APIs,
binary reproducible Gibbs transitions, surrogate gradients, and dataset
loading.

## Spin and RBM structure figures

Run `uv run python scripts/plot_rbm.py` after training. It writes
`evidence/mnist-spin-configurations.png` and
`evidence/rbm-visible-hidden-couplings.png`. A binarized pixel `vᵢ ∈ {0,1}`
is represented as an Ising spin `sᵢ = 2vᵢ - 1 ∈ {-1,+1}`: black pixels are
−1 and white pixels are +1. The RBM diagram shows the 784 visible and 128
hidden binary units, their dense bipartite connectivity, learned CD-1 weight
matrix `W` and weight distribution, plus visible and hidden bias ranges. The
784×128 matrix heatmap is the full learned coupling set; lines in the schematic
only indicate dense connectivity, not individual weight values.

`scripts/plot_rbm_training.py` reads `artifacts/rbm/metrics.json` and writes
`evidence/rbm-training.png`, comparing the epoch-mean CD surrogate and held-out
one-step changed-pixel fraction for CD-1 and CD-10. The surrogate is recorded
after each epoch as the mean of that epoch's batch values; reconstruction error
is measured at initialization and after each epoch with fixed evaluation
randomness.

## Frozen-model Gibbs sampling comparison

After both parameter files exist, run:

```sh
uv run python scripts/compare_rbm_sampling.py
```

It freezes CD-1/CD-10 parameters, starts chains
from eight held-out test digits and eight matched uniform-random binary visible
configurations (the CLI supports changing the number), and runs eight chains
per start. Identical starts and PRNG keys are used for both models. The retained
visible states at 0, 1, 10, 100, and 1000 steps are stored locally in ignored
`artifacts/rbm/sampling-chains.npz`, with settings in
`artifacts/rbm/sampling-settings.json`.

The figures `evidence/rbm-heldout-sampling.png` and
`evidence/rbm-random-sampling.png` show one representative chain at each
checkpoint for both frozen models; `evidence/rbm-heldout-reconstructions.png`
compares eight held-out inputs with their 1000-step outputs. They are binary
visible configurations (white = 1, black = 0). These samples are not clamped to
the input after initialization. In these illustrative chains CD-1 becomes
nearly blank by 1000 steps, while CD-10 retains more structure but does not
reproduce the original digits reliably. This small visualization is qualitative,
not a quantitative measure of mixing or sample quality.

## Repository conventions

- Use `uv` for Python environment and package management; CPU is the default.
- Save generated figures in `evidence/`.
- Save raw runs and their settings in `artifacts/`.
- Keep downloads and raw run outputs out of git.
