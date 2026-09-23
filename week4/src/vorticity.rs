//! Two-dimensional incompressible flow in vorticity form on a periodic
//! `[0, 2pi)^2` grid.
//!
//! We integrate
//!
//! ```text
//!   d omega / dt = -(u . grad) omega + nu Laplacian(omega)
//! ```
//!
//! with `omega = d v/dx - d u/dy`, `u = d psi/dy`, `v = -d psi/dx` and
//! `Laplacian(psi) = -omega`. Derivatives are pseudospectral, products are
//! evaluated on the grid, and the two-thirds rule keeps only
//! `|kx|, |ky| <= floor(n/3)` in the vorticity and in every product.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct Grid2d {
    n: usize,
    kmax: usize,
}

impl Grid2d {
    pub fn new(n: usize) -> Self {
        assert!(n >= 6, "grid needs at least six points per side");
        Self { n, kmax: n / 3 }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn len(&self) -> usize {
        self.n * self.n
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn dealias_kmax(&self) -> usize {
        self.kmax
    }

    pub fn flat(&self, iy: usize, ix: usize) -> usize {
        iy * self.n + ix
    }

    pub fn kx(&self, ix: usize) -> f64 {
        wavenumber(ix, self.n)
    }

    pub fn ky(&self, iy: usize) -> f64 {
        wavenumber(iy, self.n)
    }

    pub fn x(&self, ix: usize) -> f64 {
        2.0 * PI * ix as f64 / self.n as f64
    }

    pub fn y(&self, iy: usize) -> f64 {
        2.0 * PI * iy as f64 / self.n as f64
    }
}

fn wavenumber(i: usize, n: usize) -> f64 {
    if i < n / 2 {
        i as f64
    } else {
        i as f64 - n as f64
    }
}

/// A spectral derivative operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivative {
    /// d/dx
    X,
    /// d^2/dx^2
    XX,
    /// d^2/dx dy
    XY,
    /// d^2/dx^2 + d^2/dy^2
    Laplacian,
}

/// Second-order centred differences with periodic wrapping, the same stencil
/// the solver's finite-difference path uses.
pub fn centred_derivative(field: &[f64], n: usize, op: Derivative) -> Vec<f64> {
    let dx = 2.0 * PI / n as f64;
    let at = |ix: isize, iy: isize| -> f64 {
        let ix = ix.rem_euclid(n as isize) as usize;
        let iy = iy.rem_euclid(n as isize) as usize;
        field[iy * n + ix]
    };
    let mut out = vec![0.0; n * n];
    for iy in 0..n {
        for ix in 0..n {
            let (i, j) = (ix as isize, iy as isize);
            out[iy * n + ix] = match op {
                Derivative::X => (at(i + 1, j) - at(i - 1, j)) / (2.0 * dx),
                Derivative::XX => (at(i + 1, j) - 2.0 * at(i, j) + at(i - 1, j)) / (dx * dx),
                Derivative::XY => {
                    (at(i + 1, j + 1) - at(i + 1, j - 1) - at(i - 1, j + 1) + at(i - 1, j - 1))
                        / (4.0 * dx * dx)
                }
                Derivative::Laplacian => {
                    (at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1)
                        - 4.0 * at(i, j))
                        / (dx * dx)
                }
            };
        }
    }
    out
}

pub struct Spectral {
    grid: Grid2d,
    nu: f64,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    keep: Vec<bool>,
    inv_k2: Vec<f64>,
}

impl Spectral {
    pub fn new(grid: Grid2d, nu: f64) -> Self {
        let n = grid.n();
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);

        let mut kx = Vec::with_capacity(n * n);
        let mut ky = Vec::with_capacity(n * n);
        let mut keep = Vec::with_capacity(n * n);
        let mut inv_k2 = Vec::with_capacity(n * n);
        let kmax = grid.dealias_kmax() as f64;
        for iy in 0..n {
            for ix in 0..n {
                let kxi = grid.kx(ix);
                let kyi = grid.ky(iy);
                kx.push(kxi);
                ky.push(kyi);
                keep.push(kxi.abs() <= kmax && kyi.abs() <= kmax);
                let k2 = kxi * kxi + kyi * kyi;
                inv_k2.push(if k2 > 0.0 { 1.0 / k2 } else { 0.0 });
            }
        }

        Self {
            grid,
            nu,
            forward,
            inverse,
            kx,
            ky,
            keep,
            inv_k2,
        }
    }

    pub fn grid(&self) -> &Grid2d {
        &self.grid
    }

    pub fn nu(&self) -> f64 {
        self.nu
    }

    fn transform(&self, buf: &mut [Complex<f64>], inverse: bool) {
        let n = self.grid.n();
        let fft = if inverse { &self.inverse } else { &self.forward };
        let mut scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];

        for iy in 0..n {
            let row = &mut buf[iy * n..(iy + 1) * n];
            fft.process_with_scratch(row, &mut scratch);
        }

        let mut column = vec![Complex::new(0.0, 0.0); n];
        for ix in 0..n {
            for iy in 0..n {
                column[iy] = buf[iy * n + ix];
            }
            fft.process_with_scratch(&mut column, &mut scratch);
            for iy in 0..n {
                buf[iy * n + ix] = column[iy];
            }
        }
    }

    pub fn forward_real(&self, field: &[f64]) -> Vec<Complex<f64>> {
        let mut buf: Vec<Complex<f64>> =
            field.iter().map(|&value| Complex::new(value, 0.0)).collect();
        self.transform(&mut buf, false);
        buf
    }

    pub fn inverse_to_complex(&self, spec: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut buf = spec.to_vec();
        self.transform(&mut buf, true);
        let scale = 1.0 / self.grid.len() as f64;
        for value in buf.iter_mut() {
            *value *= scale;
        }
        buf
    }

    pub fn inverse_real(&self, spec: &[Complex<f64>]) -> Vec<f64> {
        self.inverse_to_complex(spec).iter().map(|c| c.re).collect()
    }

    fn truncate(&self, spec: &mut [Complex<f64>]) {
        for (value, keep) in spec.iter_mut().zip(&self.keep) {
            if !keep {
                *value = Complex::new(0.0, 0.0);
            }
        }
    }

    /// Velocity from a vorticity spectrum, `u = d psi/dy`, `v = -d psi/dx`.
    pub fn velocity(&self, omega_hat: &[Complex<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n = self.grid.len();
        let i_unit = Complex::new(0.0, 1.0);
        let mut u_hat = vec![Complex::new(0.0, 0.0); n];
        let mut v_hat = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            let psi = omega_hat[i] * self.inv_k2[i];
            u_hat[i] = i_unit * self.ky[i] * psi;
            v_hat[i] = -i_unit * self.kx[i] * psi;
        }
        self.truncate(&mut u_hat);
        self.truncate(&mut v_hat);
        (self.inverse_real(&u_hat), self.inverse_real(&v_hat))
    }

    /// Vorticity spectrum from velocities: `omega_hat = i kx v_hat - i ky u_hat`.
    pub fn vorticity_hat(&self, u: &[f64], v: &[f64]) -> Vec<Complex<f64>> {
        let u_hat = self.forward_real(u);
        let v_hat = self.forward_real(v);
        let i_unit = Complex::new(0.0, 1.0);
        let mut omega_hat: Vec<Complex<f64>> = (0..self.grid.len())
            .map(|i| i_unit * self.kx[i] * v_hat[i] - i_unit * self.ky[i] * u_hat[i])
            .collect();
        self.truncate(&mut omega_hat);
        omega_hat
    }

    pub fn physical_omega(&self, omega_hat: &[Complex<f64>]) -> Vec<f64> {
        self.inverse_real(omega_hat)
    }

    /// Derivative of a grid field by its Fourier symbol.
    pub fn differentiate(&self, field: &[f64], op: Derivative) -> Vec<f64> {
        let mut spec = self.forward_real(field);
        for i in 0..self.grid.len() {
            let (kx, ky) = (self.kx[i], self.ky[i]);
            let factor = match op {
                Derivative::X => Complex::new(0.0, kx),
                Derivative::XX => Complex::new(-kx * kx, 0.0),
                Derivative::XY => Complex::new(-kx * ky, 0.0),
                Derivative::Laplacian => Complex::new(-(kx * kx + ky * ky), 0.0),
            };
            spec[i] *= factor;
        }
        self.truncate(&mut spec);
        self.inverse_real(&spec)
    }

    /// `E = 0.5 * mean(u^2 + v^2)`.
    pub fn energy(&self, u: &[f64], v: &[f64]) -> f64 {
        let n = self.grid.len() as f64;
        let sum: f64 = u
            .iter()
            .zip(v)
            .map(|(a, b)| a * a + b * b)
            .sum();
        0.5 * sum / n
    }

    /// `Z = 0.5 * mean(omega^2)`.
    pub fn enstrophy(&self, omega: &[f64]) -> f64 {
        let n = self.grid.len() as f64;
        let sum: f64 = omega.iter().map(|w| w * w).sum();
        0.5 * sum / n
    }

    pub fn state_from_hat(&self, omega_hat: &[Complex<f64>]) -> Vec<f64> {
        let mut state = vec![0.0; 2 * self.grid.len()];
        for (i, value) in omega_hat.iter().enumerate() {
            state[2 * i] = value.re;
            state[2 * i + 1] = value.im;
        }
        state
    }

    pub fn hat_from_state(&self, state: &[f64]) -> Vec<Complex<f64>> {
        (0..self.grid.len())
            .map(|i| Complex::new(state[2 * i], state[2 * i + 1]))
            .collect()
    }

    /// Right-hand side `-(u . grad) omega + nu Laplacian(omega)` on the
    /// interleaved spectrum state `[re, im, re, im, ...]`.
    pub fn rhs(&self, state: &[f64]) -> Vec<f64> {
        let n = self.grid.len();
        let i_unit = Complex::new(0.0, 1.0);

        let mut omega = self.hat_from_state(state);
        self.truncate(&mut omega);

        let (u, v) = self.velocity(&omega);

        let mut wx_hat = vec![Complex::new(0.0, 0.0); n];
        let mut wy_hat = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            wx_hat[i] = i_unit * self.kx[i] * omega[i];
            wy_hat[i] = i_unit * self.ky[i] * omega[i];
        }
        self.truncate(&mut wx_hat);
        self.truncate(&mut wy_hat);
        let wx = self.inverse_real(&wx_hat);
        let wy = self.inverse_real(&wy_hat);

        let advection: Vec<f64> = (0..n).map(|i| u[i] * wx[i] + v[i] * wy[i]).collect();
        let mut rhs_hat = self.forward_real(&advection);
        self.truncate(&mut rhs_hat);

        let mut out = vec![0.0; 2 * n];
        for i in 0..n {
            let k2 = self.kx[i] * self.kx[i] + self.ky[i] * self.ky[i];
            let value = -rhs_hat[i] - omega[i] * (self.nu * k2);
            out[2 * i] = value.re;
            out[2 * i + 1] = value.im;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_is_divergence_free() {
        let grid = Grid2d::new(24);
        let spec = Spectral::new(grid, 0.0);
        let n = spec.grid().n();
        let mut omega = vec![0.0; spec.grid().len()];
        for iy in 0..n {
            for ix in 0..n {
                omega[spec.grid().flat(iy, ix)] =
                    (grid.x(ix)).cos() * (grid.y(iy)).cos();
            }
        }
        let omega_hat = spec.forward_real(&omega);
        let (u, v) = spec.velocity(&omega_hat);
        let u_hat = spec.forward_real(&u);
        let v_hat = spec.forward_real(&v);
        let mut worst = 0.0_f64;
        for i in 0..spec.grid().len() {
            let div = spec.kx[i] * u_hat[i] + spec.ky[i] * v_hat[i];
            worst = worst.max(div.norm());
        }
        assert!(worst < 1e-9, "divergence {worst}");
    }

    #[test]
    fn velocity_reproduces_its_vorticity() {
        let grid = Grid2d::new(24);
        let spec = Spectral::new(grid, 0.0);
        let mut omega = vec![0.0; spec.grid().len()];
        for iy in 0..grid.n() {
            for ix in 0..grid.n() {
                omega[grid.flat(iy, ix)] = (grid.x(ix)).sin() * (grid.y(iy)).cos();
            }
        }
        let omega_hat = spec.forward_real(&omega);
        let (u, v) = spec.velocity(&omega_hat);
        let recovered = spec.physical_omega(&spec.vorticity_hat(&u, &v));
        let worst = omega
            .iter()
            .zip(&recovered)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-9, "vorticity mismatch {worst}");
    }

    #[test]
    fn two_thirds_rule_sets_the_limit() {
        let grid = Grid2d::new(64);
        assert_eq!(grid.dealias_kmax(), 21);
        let grid = Grid2d::new(24);
        assert_eq!(grid.dealias_kmax(), 8);
    }

    #[test]
    fn out_of_band_modes_are_projected_out() {
        let grid = Grid2d::new(24);
        let spec = Spectral::new(grid, 0.0);
        let kmax = grid.dealias_kmax() as i64;
        let beyond = kmax + 1;

        let mut omega = vec![0.0; grid.len()];
        for iy in 0..grid.n() {
            for ix in 0..grid.n() {
                let angle = beyond as f64 * grid.x(ix) + beyond as f64 * grid.y(iy);
                omega[grid.flat(iy, ix)] = angle.cos();
            }
        }
        let omega_hat = spec.forward_real(&omega);
        let state = spec.state_from_hat(&omega_hat);
        let rhs = spec.rhs(&state);
        let worst = rhs.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(worst < 1e-12, "mode beyond the band leaked: {worst}");
    }

    #[test]
    fn taylor_green_has_vanishing_advection() {
        let grid = Grid2d::new(24);
        let spec = Spectral::new(grid, 0.0);
        let mut u = vec![0.0; grid.len()];
        let mut v = vec![0.0; grid.len()];
        for iy in 0..grid.n() {
            for ix in 0..grid.n() {
                let (x, y) = (grid.x(ix), grid.y(iy));
                let idx = grid.flat(iy, ix);
                u[idx] = x.cos() * y.sin();
                v[idx] = -x.sin() * y.cos();
            }
        }
        let omega_hat = spec.vorticity_hat(&u, &v);
        let state = spec.state_from_hat(&omega_hat);
        let rhs = spec.rhs(&state);
        let worst = rhs.iter().map(|value| value.abs()).fold(0.0, f64::max);
        assert!(worst < 1e-9, "advection should vanish, got {worst}");
    }

    #[test]
    fn dissipation_decays_a_single_mode() {
        let grid = Grid2d::new(24);
        let nu = 0.3;
        let spec = Spectral::new(grid, nu);
        let mut omega = vec![0.0; grid.len()];
        for iy in 0..grid.n() {
            for ix in 0..grid.n() {
                omega[grid.flat(iy, ix)] = (grid.x(ix)).cos() * (grid.y(iy)).cos();
            }
        }
        let omega_hat = spec.forward_real(&omega);
        let state = spec.state_from_hat(&omega_hat);
        let rhs = spec.rhs(&state);
        let derivative = spec.hat_from_state(&rhs);
        let mut worst = 0.0_f64;
        for i in 0..grid.len() {
            let k2 = spec.kx[i] * spec.kx[i] + spec.ky[i] * spec.ky[i];
            let expected = -nu * k2 * omega_hat[i];
            worst = worst.max((derivative[i] - expected).norm());
        }
        assert!(worst < 1e-9, "dissipation mismatch {worst}");
    }

    #[test]
    fn centred_derivatives_match_analytic_on_a_smooth_mode() {
        let n = 32;
        let grid = Grid2d::new(n);
        let mut values = vec![0.0; grid.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (grid.x(ix), grid.y(iy));
                values[grid.flat(iy, ix)] = (3.0 * x).sin() * (2.0 * y).cos();
            }
        }
        let lap = centred_derivative(&values, n, Derivative::Laplacian);
        let worst = (0..n)
            .flat_map(|iy| (0..n).map(move |ix| (iy, ix)))
            .map(|(iy, ix)| {
                let (x, y) = (grid.x(ix), grid.y(iy));
                let exact = -13.0 * (3.0 * x).sin() * (2.0 * y).cos();
                (lap[grid.flat(iy, ix)] - exact).abs()
            })
            .fold(0.0, f64::max);
        assert!(worst < 0.4, "centred Laplacian error {worst}");
    }

    #[test]
    fn spectral_derivatives_are_exact_on_resolved_modes() {
        let n = 32;
        let grid = Grid2d::new(n);
        let spec = Spectral::new(grid, 0.0);
        let mut values = vec![0.0; grid.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (grid.x(ix), grid.y(iy));
                values[grid.flat(iy, ix)] = (3.0 * x).sin() * (2.0 * y).cos();
            }
        }
        for op in [
            Derivative::X,
            Derivative::XX,
            Derivative::XY,
            Derivative::Laplacian,
        ] {
            let computed = spec.differentiate(&values, op);
            let worst = (0..n)
                .flat_map(|iy| (0..n).map(move |ix| (iy, ix)))
                .map(|(iy, ix)| {
                    let (x, y) = (grid.x(ix), grid.y(iy));
                    let (s3x, c3x) = ((3.0 * x).sin(), (3.0 * x).cos());
                    let (s2y, c2y) = ((2.0 * y).sin(), (2.0 * y).cos());
                    let exact = match op {
                        Derivative::X => 3.0 * c3x * c2y,
                        Derivative::XX => -9.0 * s3x * c2y,
                        Derivative::XY => -6.0 * c3x * s2y,
                        Derivative::Laplacian => -13.0 * s3x * c2y,
                    };
                    (computed[grid.flat(iy, ix)] - exact).abs()
                })
                .fold(0.0, f64::max);
            assert!(worst < 1e-12, "{op:?} error {worst}");
        }
    }

    #[test]
    fn integrators_reproduce_the_taylor_green_decay() {
        use crate::integrators::{Integrator, RungeKutta4};
        let grid = Grid2d::new(32);
        let nu = 0.1;
        let spec = Spectral::new(grid, nu);

        let mut u = vec![0.0; grid.len()];
        let mut v = vec![0.0; grid.len()];
        for iy in 0..grid.n() {
            for ix in 0..grid.n() {
                let (x, y) = (grid.x(ix), grid.y(iy));
                let idx = grid.flat(iy, ix);
                u[idx] = x.cos() * y.sin();
                v[idx] = -x.sin() * y.cos();
            }
        }
        let omega_hat = spec.vorticity_hat(&u, &v);
        let state = spec.state_from_hat(&omega_hat);

        let dt = 0.005;
        let steps = 200;
        let final_state = RungeKutta4.advance(&state, dt, steps, |s| spec.rhs(s));
        let t = steps as f64 * dt;

        let (u_end, v_end) = spec.velocity(&spec.hat_from_state(&final_state));
        let energy = spec.energy(&u_end, &v_end);
        let exact = 0.25 * (-4.0 * nu * t).exp();
        assert!(
            (energy - exact).abs() < 1e-6,
            "E = {energy}, exact {exact}"
        );
    }
}
