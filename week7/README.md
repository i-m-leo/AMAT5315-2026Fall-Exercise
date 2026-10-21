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

Animate the configured checkpoints after training:

```sh
JAX_PLATFORMS=cpu uv run python scripts/movie_training_progress.py
```

`evidence/training-progress.gif` has one stage per configured update. Each stage
shows ten complete DDPM reverse paths and the loss history through that update.
The same ten initial Gaussian samples and reverse-step noises are reused at
every checkpoint for a paired comparison. Coordinate limits adapt per stage to
keep even the untrained model's large outputs visible; the current scale is
labeled. The paths, shared random inputs, and movie settings are retained under
ignored `artifacts/training-progress/`.

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

For a 2000-sample, 1000-step reverse-process movie and per-mode evaluation, run:

```sh
JAX_PLATFORMS=cpu uv run python scripts/movie_reverse_diffusion.py
```

The movie `evidence/ddpm-reverse-process.gif` animates the generated points
from x₁₀₀₀ to x₀ beside a fixed target sample. The final comparison figure
`evidence/ddpm-final-comparison.png` shows both distributions and a table of
nearest-centre generated-mode population, empirical centre, and per-coordinate
population standard deviation. The full trajectory, settings, target reference,
and per-mode counts, centres, variances, and standard deviations are retained
under ignored `artifacts/reverse-diffusion/`.

Downloads and raw runs in `inputs/` and `artifacts/` are git-ignored. Evidence
figures, source, tests, the lockfile, and the exercise configuration are
committed.
