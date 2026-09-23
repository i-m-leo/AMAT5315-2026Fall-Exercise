//! Spectral rate function `u_t = -c u_x + nu u_xx` on a periodic grid.

use crate::grid::PeriodicGrid;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

pub struct FourierRhs {
    grid: PeriodicGrid,
    c: f64,
    nu: f64,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    symbol: Vec<Complex<f64>>,
}

impl FourierRhs {
    pub fn new(grid: PeriodicGrid, c: f64, nu: f64) -> Self {
        let n = grid.n();
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let symbol = grid
            .wavenumbers()
            .into_iter()
            .map(|k| {
                let ik = Complex::new(0.0, k);
                -Complex::new(c, 0.0) * ik - Complex::new(nu * k * k, 0.0)
            })
            .collect();
        Self {
            grid,
            c,
            nu,
            forward,
            inverse,
            symbol,
        }
    }

    pub fn grid(&self) -> &PeriodicGrid {
        &self.grid
    }

    pub fn rate(&self, state: &[f64]) -> Vec<f64> {
        let n = self.grid.n();
        let mut buffer: Vec<Complex<f64>> = state
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        self.forward.process(&mut buffer);
        for (value, symbol) in buffer.iter_mut().zip(&self.symbol) {
            *value *= symbol;
        }
        self.inverse.process(&mut buffer);
        let scale = 1.0 / n as f64;
        buffer.iter().map(|value| value.re * scale).collect()
    }

    pub fn advection_speed(&self) -> f64 {
        self.c
    }

    pub fn diffusivity(&self) -> f64 {
        self.nu
    }

    /// Exact solution of the periodic linear problem at time `t`, obtained by
    /// evolving each Fourier mode with its symbol: `exp(symbol * t)`.
    pub fn exact(&self, state: &[f64], t: f64) -> Vec<f64> {
        let n = self.grid.n();
        let mut buffer: Vec<Complex<f64>> = state
            .iter()
            .map(|&value| Complex::new(value, 0.0))
            .collect();
        self.forward.process(&mut buffer);
        for (value, symbol) in buffer.iter_mut().zip(&self.symbol) {
            *value *= (symbol * t).exp();
        }
        self.inverse.process(&mut buffer);
        let scale = 1.0 / n as f64;
        buffer.iter().map(|value| value.re * scale).collect()
    }
}
