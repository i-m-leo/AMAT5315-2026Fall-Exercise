# Week 5 regeneration guide

This guide rebuilds every committed file in `week5/` from a clean clone, and
reproduces the reference forward run. Run all commands from `week5/`.

The week has two parts: by-hand and JAX automatic differentiation of a
Lennard-Jones cluster (the committed `artifacts/ad/`), and the `seismic`
crate, which simulates acoustic waves in a heterogeneous model and
differentiates its timestep with Enzyme. All three modes — forward, born,
and adjoint — are implemented.

## Requirements

- Rust and Cargo, with the pinned `nightly-2026-09-05` toolchain and its
  `enzyme` component (`rust-toolchain.toml` selects both; `rustup toolchain
  install nightly-2026-09-05 --profile minimal --component enzyme`)
- Python 3.10 or newer with `jax`, `numpy`, and `matplotlib`

The plotting scripts set `MPLCONFIGDIR` to a scratch directory; any writable
path works, and `/tmp/mplconfig-week5` is used below.

## Automatic differentiation

By-hand forward and reverse mode on the straight-line program
`U = 4 (r^-12 - r^-6)`, checked against `jax.grad`:

```sh
python3 scripts/ad_hand.py
```

Forward and reverse mode against centred finite differences, plus the
computation graphs from `jax.make_jaxpr`:

```sh
python3 scripts/modes_compare.py
python3 scripts/draw_graphs.py
```

Gradient timing versus cluster size (forward one JVP per input, reverse one
VJP in total):

```sh
python3 scripts/scaling_ad.py
```

These write `artifacts/ad/derivatives.json`, `modes.png`, `graph.png`,
`grad-graph.png`, and `scaling.png`.

## The experiment inputs

The two panels of `artifacts/inputs.png` show what `inputs/reflector.json`
defines: the background speed, the perturbation, the shots, and the
receivers in kilometres, and the Ricker pulse at every step's start against
time in seconds.

```sh
MPLCONFIGDIR=/tmp/mplconfig-week5 python3 scripts/plot_inputs.py
```

## The seismic crate

The solver integrates `u_tt + 2 sigma u_t = c^2 lap(u) + q` with centred
second differences in time and the five-point Laplacian in space. The domain
starts at rest with a zero boundary ring; a quadratic sponge
`sigma = strength (1 - d/width)^2`, with `d` the distance to the nearest edge
in cells, absorbs waves near the boundary. The source is a Ricker pulse
`(1 - 2a) exp(-a)`, `a = (pi f0 (t - t0))^2`, spread over a unit-peak
Gaussian footprint. Receivers sample `u^{n+1}` after every update, the first
shot is recorded at every `--every` steps, and `echo.npy` repeats that shot
with `background + perturbation` and subtracts the background recording.

```sh
cargo build --release
cargo test
```

The integration test builds the binary from `CARGO_BIN_EXE_seismic`, runs it
on a tiny generated experiment, and checks the artifacts, the `.npy`
container, the CLI errors, and — when the untracked `inputs/reflector.json`
is present — the reference per-shot L2 norms.

**The reference run.** With `inputs/reflector.json` in place:

```sh
cargo run --release -- --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
```

This prints

```text
shot	mode	l2
0	forward	6.623960807913e0
1	forward	6.798645121807e0
2	forward	6.623960807913e0
```

and writes `run.json`, `result.json`, `traces.npy` (float64
`[shot, step, receiver]`), `wavefield.npy` and `echo.npy` (float32
`[frame, z, x]`, 80 frames at steps 3, 6, ..., 240). Compare against the
course reference numbers, then plot:

```sh
python3 scripts/check_forward.py
MPLCONFIGDIR=/tmp/mplconfig-week5 python3 scripts/plot_forward.py
```

`check_forward.py` verifies the per-shot L2 norms, the trace norm
11.574769503614014, the trace peaks, the recording schedule, and the echo's
reflection at the layer depth; it writes `artifacts/forward/verify.txt`.
`plot_forward.py` writes `gathers.png` and `wavefield.png`.

**Born.** The Born approximation holds the source, damping, and initial
state fixed and propagates the experiment's velocity perturbation through
the background wavefield, chaining Enzyme's timestep JVPs:

```sh
cargo run --release -- --experiment inputs/reflector.json --mode born --out artifacts/born
```

This writes `born_data.npy` (float64 `[shot, step, receiver]`); its norm is
0.18667589618149358.

**Adjoint.** The migrated image stores the full trajectory, then chains the
timestep VJPs backward from step 239 to 0, back-propagating the receiver
weights (`--data`, the Born data) and summing the velocity sensitivity over
time and shots:

```sh
cargo run --release -- --experiment inputs/reflector.json --mode adjoint \
  --data artifacts/born/born_data.npy --out artifacts/adjoint
```

This writes `image.npy` (float64 `[z, x]`). The Born data `v` and the image
`K^T K v` satisfy `<K v, K v> = <v, K^T K v>`: both sides are
0.0348478902151638, equal to within 1e-15, and the image's depth profile
peaks at z = 21, the reflector. `result.json` gains a `statistics` object
(`storage`, `reverse_calls`, `scheduler_forward_calls`, `peak_saved_states`,
`per_shot`).

```sh
python3 scripts/check_born_adjoint.py
MPLCONFIGDIR=/tmp/mplconfig-week5 python3 scripts/plot_born_adjoint.py
```

`check_born_adjoint.py` verifies the Born norm, the dot-product identity, the
reflector depth, and the statistics, and writes
`artifacts/adjoint/verify.txt`; `plot_born_adjoint.py` writes
`artifacts/born/born.png` and `artifacts/adjoint/image.png`.

**Enzyme setup.** The differentiated timestep lives in `src/kernel.rs`, a
`no_std` static library that `build.rs` compiles with `-Zautodiff=Enable` and
links into the crate; the autodiff pass never sees the CLI, JSON, or
allocation. `src/enzyme.rs` declares the C ABI wrappers and checks the slice
lengths before every call. `cargo test` includes the checks that the JVP
matches a central finite difference and that the VJP is the exact transpose
of the JVP; no hand-written tangent or adjoint is used anywhere.

The input JSON is the course's `week5-seismic-experiment-v1` schema and is
not committed (`inputs/` is ignored); `inputs/reflector.json` and the
Marmousi model come with the course materials.
