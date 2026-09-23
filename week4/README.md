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
