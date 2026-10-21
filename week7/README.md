# Week 7: Gaussian-mixture diffusion in JAX

The target is the equally weighted mixture

\[
p(x_0)=\tfrac14\sum_{k=1}^4\mathcal N(x_0;\mu_k,0.09I),\qquad
\mu_k\in\{(3,0),(0,3),(-3,0),(0,-3)\}.
\]

The linear schedule has 1000 per-step variances,
`beta_t = linspace(0.0001, 0.02, 1000)`. The forward transition is

\[
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
\]

`scripts/diffusion.py` implements target sampling, forward noising, a
sinusoidal time-conditioned SiLU MLP, and DDPM reverse sampling. The MLP
architecture, time features, Adam settings, and checkpoint updates are read
from `diffusion.toml`. JAX uses CPU by default; an explicit `JAX_PLATFORMS`
setting can override it.

## Setup and tests

From `week7/`:

```sh
uv sync --extra test
uv run pytest
```

## Forward process

```sh
JAX_PLATFORMS=cpu uv run python scripts/forward_diffusion.py
```

This saves the full trajectory and its settings in ignored
`artifacts/forward-diffusion/`, and writes
`evidence/forward-diffusion.png` at levels 0, 1, 10, 100, 500, and 1000.

Animate that same trajectory (each point follows the same sample across levels)
and print the terminal coordinate statistics with:

```sh
uv run python scripts/movie_forward_diffusion.py
```

The GIF `evidence/forward-diffusion-modes.gif` uses fixed axes from -6 to 6.
The t=1000 population means and variances are printed and also saved with the
movie settings under ignored `artifacts/forward-diffusion/`.

## Train the noise predictor

```sh
JAX_PLATFORMS=cpu uv run python scripts/train_diffusion.py \
  --config diffusion.toml --out artifacts/training
```

At every update, the script draws fresh target examples, uniformly draws
integer noise levels 1 through 1000, adds Gaussian noise using the closed-form
forward marginal, and minimizes mean squared error between the sampled and
predicted noise. It uses the configured cosine-decayed Adam optimizer. The
training output includes a copy of `diffusion.toml`, effective `settings.json`,
per-update `loss-history.csv`, and model checkpoints at every configured update.
The update-0 checkpoint is saved before any optimization. Plot the recorded loss
history with:

```sh
uv run python scripts/plot_training.py
```

This saves `evidence/noise-prediction-loss.png`.

## DDPM reverse sampling

After training:

```sh
JAX_PLATFORMS=cpu uv run python scripts/sample_diffusion.py \
  --config diffusion.toml --checkpoint artifacts/training/checkpoint-update-25000.npz
```

The sampler starts from standard Gaussian noise and uses the learned noise
predictor in the DDPM reverse mean. It uses reverse-step variance `beta_t`,
does not clip predicted samples, and adds no random noise at the final step.
Generated samples and run settings are saved in ignored `artifacts/sampling/`;
`evidence/ddpm-samples.png` compares generated samples with the target.

Downloads and raw runs in `inputs/` and `artifacts/` are git-ignored. Evidence
figures, source, tests, the lockfile, and the exercise configuration are
committed.
