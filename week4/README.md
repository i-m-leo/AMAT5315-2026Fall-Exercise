# Week 4 regeneration guide

This guide rebuilds every committed file in `week4/` from a clean clone. Run all
commands from `week4/`.

The week has two parts: a small Rust library in `src/` with one `Integrator`
trait shared by three explicit schemes, used for a one-dimensional
advection-diffusion study, and the `field` / `fluid` pair that integrates the
two-dimensional incompressible vorticity equation with the same integrators.

## Requirements

- Rust and Cargo (edition 2021; the committed `Cargo.lock` pins `rustfft`,
  `clap`, `rand`, `serde`, and `serde_json`)
- Python 3.10 or newer with `numpy` and `matplotlib`, for the plotting scripts

The plotting commands set `MPLCONFIGDIR` to a scratch directory. Any writable
path works; `/tmp/mplconfig-week4` is used below.

## Install

```sh
cargo build --release
```

`cargo test` runs the library tests, including the vorticity checks used by the
two tools.

```sh
cargo test
```

## The two pipelines

Both pipelines read and write the same JSON shape: `case`, `n`, `seed`,
`k_band`, then `u` and `v` as `n*n` arrays in row-major order with row = y and
column = x.

**Taylor-Green.** The exact solution decays as `exp(-2 nu t)`. This is both the
initial condition and, at `--t`, the field the solver is compared with.

```sh
cargo run --release --bin field -- taylor-green --n 64 --nu 0.1 --t 1
```

**Random.** Equal-amplitude vorticity modes with `k-min <= |k| <= k-max`, one
uniform phase per mode, rescaled so `E(0) = 0.5`.

```sh
cargo run --release --bin field -- random --n 128 --seed 2026 --k-min 2 --k-max 6
```

**Solver.** `fluid` reads that JSON on stdin, takes the initial vorticity as the
spectral curl of the piped velocity, and integrates with `--method` (`euler`,
`rk2`, or `rk4`), pseudospectral derivatives, and the two-thirds rule.

```sh
cargo run --release --bin field -- taylor-green --n 64 \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green
```

`scripts/perturb.rs` inserts between them to add a vorticity ripple:

```sh
cargo run --release --bin field -- taylor-green --n 64 \
  | cargo run --release --bin perturb \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green
```

## Evidence, in production order

Each entry lists the script, the command, and the files it writes. Every path
under `evidence/` is committed.

### 1. Wave check

`scripts/wave_check.rs` runs each scheme against the exact single wave for both
rate functions.

```sh
cargo run --release --bin wave-check > evidence/wave-check.txt
```

Writes: `evidence/wave-check.txt`.

### 2. Differentiation accuracy

`scripts/differentiate.rs` compares the solver's spectral derivatives and the
periodic centred stencil against the analytic derivatives of
`g(x, y) = sin(3x) cos(2y)`.

```sh
cargo run --release --quiet --bin differentiate > evidence/differentiation.txt
```

Writes: `evidence/differentiation.txt`.

### 3. Stability plane and pulse

`scripts/dump_stability.rs` measures the per-step growth factor on the complex
plane `z = lambda h`. `scripts/pulse_field.rs` records `u(x, t)` for a periodic
Gaussian pulse under RK4.

```sh
cargo run --release --quiet --bin dump-stability -- 3.5 321 | gzip -9 > evidence/stability-grid.txt.gz
cargo run --release --quiet --bin pulse-field -- 0.045 6 > evidence/pulse-dt0.045.txt
cargo run --release --quiet --bin pulse-field -- 0.056 6 > evidence/pulse-dt0.056.txt
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_line_stability.py
```

Writes: `evidence/stability-grid.txt.gz`, `evidence/pulse-dt0.045.txt`,
`evidence/pulse-dt0.056.txt`, `evidence/line-stability.png`.

The `--release` flag matters for `pulse-field`: on the debug build the pulse
integration is slow enough to be inconvenient.

### 4. Accuracy after one lap, and method convergence

`scripts/accuracy_run.rs` laps a Gaussian pulse three ways and reports the
maximum error of each; `scripts/convergence.rs` is its supporting sweep;
`scripts/method_convergence.rs` compares the four one-step methods.
`scripts/draw_accuracy.py` draws both datasets in one figure, so both data files
must exist before it runs.

```sh
cargo run --release --quiet --bin accuracy-run -- evidence
cargo run --release --quiet --bin method-convergence -- evidence
cargo run --release --quiet --bin convergence > evidence/line-accuracy-convergence.txt
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_accuracy.py
```

Writes: `evidence/line-accuracy.txt`, `evidence/line-accuracy-profiles.txt`,
`evidence/method-convergence.txt`, `evidence/line-accuracy-convergence.txt`,
`evidence/line-accuracy.png`.

`draw_accuracy.py` reads `line-accuracy-profiles.txt` and
`method-convergence.txt`, so it must run after both of those are written. It
does not read `line-accuracy-convergence.txt`.

### 5. Taylor-Green comparison

`scripts/compare_taylor_green.py` compares the last frame with the exact field
and draws the vorticity with velocity arrows. It needs the artifacts first.

```sh
mkdir -p artifacts/taylor-green
cargo run --release --bin field -- taylor-green --n 64 \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green
cargo run --release --bin field -- taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/compare_taylor_green.py
```

Writes: `evidence/taylor-green.png`.

### 6. Stability scan and blow-up

`scripts/run_scan.sh` runs Taylor-Green and the random flow until each either
reaches its final time or diverges, then `scripts/draw_blowup.py` draws the
energies.

```sh
sh scripts/run_scan.sh
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_blowup.py
```

Writes: `evidence/blowup.png`.

### 7. Sensitivity to an initial vorticity ripple

`scripts/run_sensitivity.sh` runs each case twice, the second with the ripple
`-7e-5 * M * cos(3x) cos(4y)`, and `scripts/draw_sensitivity.py` plots the
separation.

```sh
sh scripts/run_sensitivity.sh
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_sensitivity.py
```

Writes: `evidence/sensitivity.png`.

### 8. Random flow snapshots

```sh
mkdir -p artifacts/random
cargo run --release --bin field -- random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_random.py
```

Writes: `evidence/random.png`.

### 9. Time-step order

`scripts/run_order.sh` runs Taylor-Green on an `8 x 8` grid at three steps and
`scripts/draw_order.py` draws the errors.

```sh
sh scripts/run_order.sh
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_order.py
```

Writes: `evidence/order.png`.

`scripts/diagnose_order.py` repeats the measurement from both the six-decimal
and full-precision frames and writes nothing.

### 10. Random-flow convergence and Richardson step choice

`scripts/run_convergence.sh` runs the four steps, `scripts/convergence_table.py`
writes the report, and `scripts/draw_convergence.py` draws the figure.

```sh
sh scripts/run_convergence.sh
python3 scripts/convergence_table.py
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_convergence.py
```

Writes: `evidence/convergence.json`, `evidence/convergence.png`.

## Evidence files and their commands

| File in `week4/evidence/` | Produced by |
| --- | --- |
| `wave-check.txt` | `cargo run --release --bin wave-check` |
| `differentiation.txt` | `cargo run --release --quiet --bin differentiate` |
| `stability-grid.txt.gz` | `cargo run --release --quiet --bin dump-stability -- 3.5 321 \| gzip -9` |
| `pulse-dt0.045.txt` | `cargo run --release --quiet --bin pulse-field -- 0.045 6` |
| `pulse-dt0.056.txt` | `cargo run --release --quiet --bin pulse-field -- 0.056 6` |
| `line-stability.png` | `python scripts/draw_line_stability.py` |
| `line-accuracy.txt` | `cargo run --release --quiet --bin accuracy-run -- evidence` |
| `line-accuracy-profiles.txt` | `cargo run --release --quiet --bin accuracy-run -- evidence` |
| `line-accuracy-convergence.txt` | `cargo run --release --quiet --bin convergence` |
| `line-accuracy.png` | `python scripts/draw_accuracy.py` |
| `method-convergence.txt` | `cargo run --release --quiet --bin method-convergence -- evidence` |
| `taylor-green.png` | `python scripts/compare_taylor_green.py` |
| `blowup.png` | `python scripts/draw_blowup.py` |
| `sensitivity.png` | `python scripts/draw_sensitivity.py` |
| `random.png` | `python scripts/draw_random.py` |
| `order.png` | `python scripts/draw_order.py` |
| `convergence.json` | `python3 scripts/convergence_table.py` |
| `convergence.png` | `python scripts/draw_convergence.py` |

## Not tracked

The runs under `artifacts/` are inputs to the plotting scripts and are not
committed, matching how `week3/artifacts/` is handled. `scripts/run_*.sh`
recreate them. `target/` is Cargo build output.
