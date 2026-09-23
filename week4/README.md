# Week 4 finite-difference integrators

A small Rust library in `src/` with one `Integrator` trait shared by three
explicit schemes, plus two rate functions for the advection-diffusion equation
`u_t + c u_x = nu u_xx` on a periodic `[0, 2pi)` grid.

## Layout

- `src/integrators.rs` - the `Integrator` trait (`step`, `advance`) and the
  forward Euler, explicit midpoint, and classical RK4 implementations, wrapped
  in a `Scheme` enum so the set can be iterated.
- `src/fourier.rs` - spectral rate function using FFT derivatives.
- `src/centred.rs` - rate function using centred finite differences.
- `src/grid.rs` - the periodic grid and the exact single-wave solution
  `u(x, t) = A exp(-nu k^2 t) cos(k (x - c t) + phi)`.
- `scripts/wave_check.rs` - runs each scheme against the exact wave for both
  rate functions.
- `scripts/dump_stability.rs` - measures the per-step growth factor on the
  complex plane `z = lambda h`.
- `scripts/pulse_field.rs` - records `u(x, t)` for a periodic Gaussian pulse
  under RK4.
- `scripts/accuracy_run.rs` - laps a Gaussian pulse three ways and reports the
  maximum error of each.
- `scripts/convergence.rs` - dt convergence sweep for those runs.
- `scripts/method_convergence.rs` - dt convergence of Euler, midpoint, RK4, and
  an equal-weight RK4.
- `scripts/perturb.rs` - adds a vorticity ripple to a field on stdin.
- `scripts/draw_random.py` - vorticity of the random run at four times.
- `scripts/run_order.sh` and `scripts/draw_order.py` - RK4 time-step order study.
- `scripts/run_convergence.sh` and `scripts/convergence_table.py` - random-flow
  time-step convergence against a fine reference.

## Field and fluid tools

`field` and `fluid` integrate the two-dimensional incompressible vorticity
equation `d omega/dt = -(u . grad) omega + nu Laplacian(omega)` on a periodic
`[0, 2pi)^2` grid, in the spirit of the Week 2 `md` crate: a generator writes a
field, and a solver integrates it with the `Integrator` trait from Part 1.
Their contracts are `field.design.toml` and `fluid.design.toml`.

`field` writes one JSON object to stdout with `case`, `n`, `seed`, `k_band`, and
`u`, `v` as `n*n` arrays in row-major order with row = y and column = x.

```sh
cargo run --release --bin field -- taylor-green --n 64 --nu 0.1 --t 1
cargo run --release --bin field -- random --n 128 --seed 2026 --k-min 2 --k-max 6
```

The `taylor-green` case is the exact solution
`u = cos(x) sin(y) exp(-2 nu t)`, `v = -sin(x) cos(y) exp(-2 nu t)`. The
`random` case sums equal-amplitude cosine modes, `k-min <= |k| <= k-max`, with
one uniform phase per mode, then rescales so `E(0) = 0.5`.

`fluid` reads that JSON on stdin, takes the initial vorticity as the spectral
curl `omega_hat = i kx v_hat - i ky u_hat`, and integrates with `--method`
(`euler`, `rk2`, or `rk4`). Derivatives are pseudospectral and the two-thirds
rule keeps only `|kx|, |ky| <= floor(n/3)` in the vorticity and in every
product.

```sh
cargo run --release --bin field -- taylor-green --n 64 \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green
```

`fluid` prints `t`, energy `E = 0.5 * mean(u^2 + v^2)`, and enstrophy
`Z = 0.5 * mean(omega^2)` per snapshot; it stops at the first non-finite energy,
prints that line, and exits 1 without storing that frame. It writes
`<out>/run.json` and `<out>/fields.jsonl` as the design files describe.

The `src/vorticity.rs` tests check that the velocity is divergence free, that it
reproduces its own vorticity, that Taylor-Green advection vanishes, that a
single mode decays at its exact spectral rate, that out-of-band modes are
projected away, and that the library integrators reproduce the exact
Taylor-Green energy decay `0.25 exp(-4 nu t)`.

## Differentiation accuracy

`scripts/differentiate.rs` applies the solver's spectral derivative operators
(`Derivative::X`, `XX`, `XY`, `Laplacian`) and the periodic second-order centred
stencil to `g(x, y) = sin(3x) cos(2y)`, and compares both against the analytic
derivatives.

```sh
cargo run --release --quiet --bin differentiate | tee evidence/differentiation.txt
```

The spectral errors are at machine precision (`~1e-14`). The centred stencil
errors fall by about a factor 4 when `dx` halves, confirming second order; the
`ratio 32/64` column is that reduction factor.

## Nyquist mode

The grid stores wavenumbers as `0, 1, .., n/2 - 1, -n/2, .., -1`, so the
Nyquist mode sits at index `n/2` with `k = -n/2`. Its first derivative vanishes
(`d/dx cos((n/2) x) = 0` at that mode), so both rate functions return zero for
it and it never travels. The centred stencil gives the same zero: at the
Nyquist mode neighbours are equal, so `(u_{j+1} - u_{j-1}) / 2dx = 0`. This is
covered by the `nyquist` tests in `src/lib.rs`.

## Run

```sh
cargo test
cargo run --bin wave-check
```

The committed output is `evidence/wave-check.txt`.

## Stability plane and pulse

`scripts/dump_stability.rs` advances `y' = lambda y` one step of `h = 1` with
each library integrator over a grid of complex `z = lambda h`, so the per-step
growth factor `|y_1|` is measured from the integrators themselves rather than
from a formula. `scripts/pulse_field.rs` records `u(x, t)` for a periodic
Gaussian pulse of standard deviation `0.35` centred at `x = pi/2`, integrated by
the library RK4. Both are drawn by `scripts/draw_line_stability.py`, which
overlays the analytic `|R(z)| = 1` curves of all three schemes and the spectral
modes of the line at `nu = 0.05`, `n = 64`, `c = 1` for `dt = 0.045` and
`dt = 0.056`.

```sh
cargo run --quiet --bin dump-stability -- 3.5 321 | gzip -9 > evidence/stability-grid.txt.gz
cargo run --quiet --release --bin pulse-field -- 0.045 6 > evidence/pulse-dt0.045.txt
cargo run --quiet --release --bin pulse-field -- 0.056 6 > evidence/pulse-dt0.056.txt
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_line_stability.py
```

Writes: `evidence/stability-grid.txt.gz`, `evidence/pulse-dt0.045.txt`,
`evidence/pulse-dt0.056.txt`, and `evidence/line-stability.png`.

At `dt = 0.045` the whole pulse spectrum is stable and the pulse advects and
diffuses smoothly. At `dt = 0.056` the Nyquist mode leaves the RK4 stability
region, so the pulse goes unstable and the field fills with grid-scale noise.

## Accuracy after one lap

`scripts/accuracy_run.rs` takes a periodic Gaussian pulse (`sigma = 0.25`,
centred at `x = pi/2`) once around the box, to `t = 2 pi`, on `n = 64`,
`c = 1`, `nu = 0.002`, in three ways: RK4 with Fourier derivatives at
`dt = 0.02`, RK4 with centred differences at `dt = 0.02`, and forward Euler with
Fourier derivatives at `dt = 0.005`. Errors are measured against the exact
periodic solution, reached by evolving the initial FFT modes with their symbol
(`FourierRhs::exact`). `scripts/draw_accuracy.py` plots the final profiles and
their pointwise error. `scripts/convergence.rs` is the supporting dt sweep.

```sh
cargo run --quiet --release --bin accuracy-run -- evidence
cargo run --quiet --release --bin convergence | tee evidence/line-accuracy-convergence.txt
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_accuracy.py
```

Writes: `evidence/line-accuracy.txt`, `evidence/line-accuracy-profiles.txt`,
`evidence/line-accuracy-convergence.txt`, and `evidence/line-accuracy.png`.

The RK4 spectral run is the accurate one (`1.80e-5`) and converges at fourth
order. Forward Euler at `dt = 0.005` is only first order and loses the pulse
amplitude (`2.10e-1`). RK4 with centred differences is worse still (`3.11e-1`):
its error does not shrink with `dt` because it is dominated by the second-order
spatial phase error of the centred stencil, not by the time integration.

## Method convergence

The third panel of `line-accuracy.png` compares the four one-step methods on a
different pulse (`sigma = 0.35`, `t = 1`, `nu = 0.05`), with Fourier
derivatives, at `dt = 0.02, 0.01, 0.005, 0.0025`. Each method's points are
fitted by a straight line in log-log space and the slope is labelled in the
legend.

```sh
cargo run --quiet --release --bin method-convergence -- evidence
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_accuracy.py
```

Writes: `evidence/method-convergence.txt` and `evidence/line-accuracy.png`.

The measured slopes are forward Euler `1.03`, explicit midpoint `2.01`,
classical RK4 `4.00`, and equal-weight RK4 `2.00`. Giving the four RK4 stages
equal weights `b = 1/4` breaks the fourth-order cancellation and drops the
scheme to second order, as the slope shows.

## Taylor-Green comparison

`scripts/compare_taylor_green.py` compares the last frame of
`artifacts/taylor-green/fields.jsonl` with the exact field
`artifacts/taylor-green/exact-t1.json`, prints the relative velocity error, and
draws the vorticity at `t = 0` and `t = 1` with velocity arrows on a shared
colour scale.

Regenerate the artifacts first (they stay out of Git, like `week3/artifacts/`):

```sh
cargo run --release --bin field -- taylor-green --n 64 \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 --out artifacts/taylor-green
cargo run --release --bin field -- taylor-green --n 64 --nu 0.1 --t 1 > artifacts/taylor-green/exact-t1.json
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/compare_taylor_green.py
```

Writes: `evidence/taylor-green.png`.

The relative velocity error is `2.41e-07`. The largest per-point difference is
exactly `1.0e-06`, which is the 6-decimal rounding that `fields.jsonl` stores,
so the true discretisation error of the run is below the recording precision.

## Stability scan and blow-up

`scripts/run_scan.sh` runs the two cases until they either reach their final
time or diverge:

- Taylor-Green, `n = 64`, `nu = 0.1`, to `t = 8`: RK4 at `dt = 0.032` and
  `0.033`.
- Random, `n = 128`, `nu = 0.004`, to `t = 10`, seed 2026, wavenumbers 2 to 6:
  RK4 at `dt = 0.040` and `0.038`, Euler at `dt = 0.01`, then RK4 at smaller
  steps until one reaches `t = 10`.

All runs snapshot every `0.5` time units and write `artifacts/scan/<name>/` plus
`artifacts/scan/<name>.tsv`. The random initial field is identical in every run.

```sh
cargo build --release
sh scripts/run_scan.sh
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_blowup.py
```

Writes: `artifacts/scan/` (not tracked) and `evidence/blowup.png`.

The largest speed of the random initial field is `2.330426`. Both specified
random RK4 steps and the Euler run diverge, so the step was reduced until
`dt = 0.033` reached `t = 10`; `dt = 0.034` is already unstable. For
Taylor-Green `dt = 0.032` survives and `dt = 0.033` diverges. The blow-up times
are marked in the figure: Taylor-Green `3.960`, random RK4 `dt = 0.038` at
`0.646`, and Euler at `0.840`.

## Sensitivity to an initial vorticity ripple

`scripts/run_sensitivity.sh` runs each case twice with RK4 at `dt = 0.01` to
`t = 20`, snapshots every `0.5`. The first run uses the original field; the
second adds the vorticity ripple

```text
delta omega(x, y) = -7e-5 * M * cos(3x) cos(4y)
```

where `M` is the largest `|u|` or `|v|` of that case's initial field
(`M = 1` for Taylor-Green, `M = 2.213145` for random). All other settings are
unchanged, so the pair differs only in that ripple. `scripts/perturb.rs` adds
the ripple through the streamfunction, so the spectral curl of the perturbed
velocity exceeds the original by exactly `delta omega` (verified to `2e-11`).

```sh
sh scripts/run_sensitivity.sh
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_sensitivity.py
```

Writes: `artifacts/sensitivity/` (not tracked) and `evidence/sensitivity.png`.

The figure plots `||omega_perturbed - omega_base||_2 / ||omega_base||_2` against
time on a log scale. The random pair grows from `2.31e-05` to `1.71e-03` (a
factor `74`), showing exponential divergence of neighbouring trajectories. The
Taylor-Green pair instead decays cleanly, from `3.50e-05` to `1.46e-07`, by the
exact factor `exp(-nu k^2 t)` for `k^2 = 3^2 + 4^2 = 25`: the flow is a single
decaying mode, so the ripple simply decays with it and there is no growth of
error.

These runs record `fields-full.jsonl` (via `fluid --full-precision`) because the
Taylor-Green separation falls below the 6-decimal rounding of the default
`fields.jsonl` from about `t = 2` onwards; the default output is unchanged.

## Random flow snapshots

`scripts/draw_random.py` draws the vorticity of the random run at `t = 0, 2, 5`,
and `10` in one row on a shared diverging colour scale.

```sh
cargo run --release --bin field -- random --n 128 --seed 2026 --k-min 2 --k-max 6 \
  | cargo run --release --bin fluid -- --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 --out artifacts/random
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_random.py
```

Writes: `evidence/random.png`.

The initially fine-grained random vorticity rolls up into coherent structures
by `t = 2` and `t = 5`, then decays under viscosity; at `t = 10` little of the
original amplitude survives, which is why that panel is faint on the shared
scale.

## Time-step order

`scripts/run_order.sh` runs Taylor-Green on an `8 x 8` grid with `nu = 0.5` to
`t = 2` using RK4 at `dt = 0.4, 0.25, 0.2`, each under
`artifacts/order/rk4-dt<dt>/`, and records the exact field
`artifacts/order/exact-t2.json`. `scripts/draw_order.py` computes the relative
L2 error of each final velocity field, plots the errors against `dt` on log-log
axes with a fitted slope, and writes `evidence/order.png`.

```sh
sh scripts/run_order.sh
MPLCONFIGDIR=/tmp/mplconfig-week4 python scripts/draw_order.py
```

Writes: `artifacts/order/` (not tracked) and `evidence/order.png`.

The three relative errors are `5.98e-04`, `8.23e-05`, and `3.37e-05`; the fitted
log-log slope is `4.16`, consistent with the fourth-order accuracy of RK4. The
three steps divide `t = 2` exactly (5, 8, and 10 steps), so no run is shortened
to land on the final time. These runs use `fluid --full-precision` so the error
is not limited by the 6-decimal recording.

## Random-flow time-step convergence

`scripts/run_convergence.sh` runs the random flow (`n = 128`, `nu = 0.004`,
`seed 2026`, wavenumbers 2 to 6) to `t = 2` with RK4 at `dt = 0.02, 0.0125,
0.01`, plus a `dt = 0.0025` reference. `scripts/convergence_table.py` compares
the final vorticity of each run against the reference and writes
`evidence/convergence.json`.

```sh
sh scripts/run_convergence.sh
python3 scripts/convergence_table.py
```

Writes: `artifacts/convergence/` (not tracked) and `evidence/convergence.json`.

```text
      dt  steps   relative error of omega
    0.02    100     2.326640e-05
  0.0125    160     3.493208e-06
    0.01    200     1.420515e-06
```

The fitted log-log slope is `4.03`, matching RK4's fourth order. The `dt = 0.0025`
reference is itself converged: halving it again to `0.00125` changes the final
field by only `5.1e-09`, some 280 times smaller than the smallest error above.
Every run keeps its final field at full precision under
`artifacts/convergence/rk4-dt<dt>/fields-full.jsonl` for the error estimate.
