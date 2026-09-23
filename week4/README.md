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
