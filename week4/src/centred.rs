//! Centred finite-difference rate function `u_t = -c u_x + nu u_xx` with
//! periodic wraps.

use crate::grid::PeriodicGrid;

pub struct CentredDifferenceRhs {
    grid: PeriodicGrid,
    c: f64,
    nu: f64,
    inv_dx: f64,
    inv_dx2: f64,
}

impl CentredDifferenceRhs {
    pub fn new(grid: PeriodicGrid, c: f64, nu: f64) -> Self {
        let dx = grid.spacing();
        Self {
            grid,
            c,
            nu,
            inv_dx: 1.0 / dx,
            inv_dx2: 1.0 / (dx * dx),
        }
    }

    pub fn grid(&self) -> &PeriodicGrid {
        &self.grid
    }

    pub fn rate(&self, state: &[f64]) -> Vec<f64> {
        let n = self.grid.n();
        (0..n)
            .map(|j| {
                let left = state[(j + n - 1) % n];
                let right = state[(j + 1) % n];
                let centre = state[j];
                let du_dx = (right - left) * 0.5 * self.inv_dx;
                let d2u_dx2 = (right - 2.0 * centre + left) * self.inv_dx2;
                -self.c * du_dx + self.nu * d2u_dx2
            })
            .collect()
    }

    pub fn advection_speed(&self) -> f64 {
        self.c
    }

    pub fn diffusivity(&self) -> f64 {
        self.nu
    }
}
