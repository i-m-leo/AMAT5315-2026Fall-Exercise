# Week 7: Gaussian-mixture forward diffusion

This exercise implements only target sampling and the forward diffusion process
in JAX. It does not train a neural network and does not implement reverse
sampling.

## Model

The target is the equally weighted mixture

\[
p(x_0)=\tfrac14\sum_{k=1}^4\mathcal N(x_0;\mu_k,0.09I),\qquad
\mu_k\in\{(3,0),(0,3),(-3,0),(0,-3)\}.
\]

The linear schedule contains 1000 per-step variances,
`beta_t = linspace(0.0001, 0.02, 1000)`. Each forward transition is

\[
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
\]

`scripts/diffusion.py` provides `sample_target`, `beta_schedule`,
`forward_diffusion` (the full trajectory including `x_0`), and `q_sample`
(the closed-form marginal at a 1-based diffusion level). JAX is configured to
use CPU by default; setting `JAX_PLATFORMS` explicitly overrides that default.

## Run

From `week7/`, install with `uv` and run tests:

```sh
uv sync --extra test
uv run pytest
```

Generate a seeded 1000-step run and visualization on CPU:

```sh
JAX_PLATFORMS=cpu uv run python scripts/forward_diffusion.py
```

The script writes the complete trajectory, beta schedule, and run settings to
ignored `artifacts/forward-diffusion/`. It saves a plot of levels 0, 1, 10,
100, 500, and 1000 to `evidence/forward-diffusion.png`. `--num-samples`,
`--seed`, `--output-dir`, and `--figure` can be used to customize the run.

Downloads and raw outputs under `inputs/` and `artifacts/` are git-ignored;
the evidence figure and source are committed.
