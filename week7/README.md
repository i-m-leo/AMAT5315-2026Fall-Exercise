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
sinusoidal time-conditioned SiLU MLP, DDPM reverse sampling, selected-step
deterministic DDIM, and the mixture's exact conditional-mean noise predictor.
The MLP architecture, time features, Adam settings, and checkpoint updates are read
from `diffusion.toml`. JAX uses CPU by default; an explicit `JAX_PLATFORMS`
setting can override it.

## Setup and tests

From `week7/`:

```sh
uv sync --extra test
JAX_PLATFORMS=cpu uv run pytest
```

`uv.lock` records the reproducible environment and CPU JAX dependency. The
reported local run used Python 3.14.7, JAX 0.12.0, NumPy 2.5.4, SciPy 1.18.1,
and safetensors 0.8.0 on CPU; the repository's starting tool/Python versions
are recorded in the root `setup.txt`. SciPy supports the unbiased energy-
distance estimator, and safetensors loads the pretrained image checkpoint.

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
The update-0 checkpoint is saved before any optimization. The configuration uses
25,000 Adam updates, batch size 512, learning rate 0.001 with cosine decay,
float32, and seed 2026. Plot the recorded loss history with:

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

## Selected-step deterministic DDIM

Compare the trained predictor with the exact conditional-mean noise predictor
using 50 evenly spaced levels by default:

```sh
JAX_PLATFORMS=cpu uv run python scripts/compare_ddim.py
```

Choose a different number of inference levels with `--inference-steps N`, or
supply an explicit descending subset beginning at 1000, for example
`--levels 1000,900,800,600,400,200,1`. Both predictors use the same initial
Gaussian samples and deterministic eta=0 DDIM updates. The oracle evaluates
`E[epsilon | x_t]` from the mixture's exact component posterior. The final
comparison figure is `evidence/ddim-comparison.png`; selected-level trajectories,
settings, and per-mode statistics are saved under ignored
`artifacts/ddim-comparison/`.

Compare full 1000-level DDIM repeats and two DDPM noise draws from the same
2000 starting Gaussian samples:

```sh
JAX_PLATFORMS=cpu uv run python scripts/compare_ddim_ddpm.py
```

`evidence/ddim-ddpm-trajectories.png` shows ten paired example paths per run;
`evidence/ddim-ddpm-samples.png` places all four sampler outputs beside 2000
exact target samples. The script prints the maximum DDIM trajectory and final-
sample differences (zero for repeated deterministic DDIM). Raw arrays and
settings are kept under ignored `artifacts/ddim-ddpm-comparison/`.

## Flow matching and the cost of fewer steps

The flow model uses a new initialization and the configuration in `flow.toml`:
three 128-wide SiLU layers, 32 sinusoidal time features scaled by `1000*s`,
25,000 Adam updates, batch size 512, learning rate 0.001 with cosine decay,
float32, and seed 2029. Training samples straight paths
`x_s=(1-s)*x0+s*z` and regresses the velocity `z-x0`.

Train and plot its recorded loss:

```sh
JAX_PLATFORMS=cpu uv run python scripts/train_flow.py \
  --config flow.toml --out artifacts/flow-training
uv run python scripts/plot_flow_training.py
```

The velocity MSE fell from 5.7001 to 2.6371. We compare the learned field
against the exact mixture marginal velocity on 4,000 marginal samples at each
time:

```sh
JAX_PLATFORMS=cpu uv run python scripts/verify_flow_velocity.py
```

The relative RMS errors at `s=0.1, 0.5, 0.9` are 0.0093, 0.0177, and 0.0651,
all below 0.1. The vector-field plot is
`evidence/flow-velocity-fields.png`; the training curve is
`evidence/flow-training-loss.png`. Both commands retain settings and numerical
results below ignored `artifacts/`.

Compare DDPM, DDIM, and Euler flow sampling on 1, 2, 5, 10, 20, 50, and 100
steps (plus 1000-level DDPM/DDIM), using 4,000 shared initial Gaussian samples
per method and 20 exact-vs-exact null pairs:

```sh
JAX_PLATFORMS=cpu uv run python scripts/compare_samplers.py
```

`evidence/sampler-step-comparison.png` plots energy distance, largest cluster-
fraction error, and largest coordinate-width error against network evaluations.
The exact-vs-exact energy-distance 95th percentile is 0.003316. At 20 steps,
flow and DDIM have energy distances 0.00221 and 0.00229 and maximum width
errors 0.0131 and 0.0254; DDPM reaches the 0.03 width tolerance at 50 steps.
Flow matching learns a continuous-time transport velocity and follows it with
Euler steps; DDIM reuses the trained noise predictor for deterministic jumps;
DDPM uses stochastic reverse steps. Coarse transport grids can fail even when
using an exact predictor, so the comparison isolates both model error and
sampling discretization. Full samples, settings, null pairs, and metrics are
stored in ignored `artifacts/sampler-comparison/`.

## Pretrained CIFAR-10 generation

Install the pinned public checkpoint and the prepared four-image test set from
`week7/`:

```sh
uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 \
  config.json diffusion_pytorch_model.safetensors \
  --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
  --local-dir inputs/cifar-model
curl -fL https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-images.npz \
  -o inputs/images.npz
```

`scripts/cifar_unet.py` implements the checkpoint's Diffusers v0.40.0
`UNet2DModel` in JAX and loads its fixed safetensors weights. Generate four
images and reverse-process stages with the 1000-step DDPM sampler:

```sh
JAX_PLATFORMS=cpu uv run python scripts/generate_cifar.py --seed 7401
```

The script does not train or modify the downloaded weights, passes model index
`t-1`, uses float32, and never clips sampling states (clipping is only for
image display). `evidence/cifar-ddpm-reverse-process.png` shows the samples at
`t=1000, 200, 50, 0`; `evidence/cifar-generated-gallery.png` and
`evidence/cifar-real-gallery.png` are native-resolution galleries rendered by
the Week 6 `image_gallery` utility. The four outputs become recognizable without
a class or text prompt. The network's photograph prior was learned before the
download; this exercise applies its noise predictions through the reverse
sampler. The selected snapshots and settings are under ignored
`artifacts/cifar-ddpm/`.

## CIFAR-10 inpainting with RePaint

Use the same fixed predictor to complete the central 20-by-20 square in each of
the four prepared test images:

```sh
JAX_PLATFORMS=cpu uv run python scripts/inpaint_cifar.py
```

RePaint visits levels `1000, 980, ..., 20, 0`, retries each jump twice except
the final jump, forward-noises known pixels to the current level, and clips only
the Tweedie clean estimate to `[-1, 1]`. Four completions per image use seeds
7201–7204. `evidence/cifar-repaint-inpainting.png` shows each truth, observation,
mean completion, pixelwise variance, and all four completions. The overall
hidden-pixel MSE is 0.1956 (the reference is about 0.24); the largest visible-
pixel change is exactly 0. Each completion takes 99 network evaluations. Full
completions, per-seed MSEs, variances, and settings are saved under ignored
`artifacts/cifar-repaint/`.

Downloaded files and raw runs in `inputs/` and `artifacts/` are git-ignored;
the pretrained model weights are not committed. Evidence figures, source,
tests, `uv.lock`, and exercise configurations are committed.
