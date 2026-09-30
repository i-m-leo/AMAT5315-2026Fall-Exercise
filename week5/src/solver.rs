//! The forward, damped acoustic solver.
//!
//! ```text
//! u_tt + 2 sigma u_t = c^2 lap(u) + q
//! ```
//!
//! discretised with centred second differences in time and the five-point
//! Laplacian in space:
//!
//! ```text
//! u^{n+1} = [2 u^n - (1 - sigma dt) u^{n-1} + dt^2 (c^2 lap(u^n) + q^n)]
//!           / (1 + sigma dt)
//! ```
//!
//! The domain starts at rest with a zero boundary ring, and a quadratic
//! sponge `sigma = strength [1 - d/width]^2` absorbs waves near the edges,
//! with `d` the distance to the nearest edge in cells.

use crate::experiment::Experiment;

/// Precomputed squared-speed model plus the sponge profile.
pub struct Model {
    nx: usize,
    c2: Vec<f64>,
    sigma: Vec<f64>,
    inv_dx2: f64,
}

impl Model {
    /// Sponge strength at one grid point (0 in the interior).
    pub fn sigma_at(&self, ix: usize, iz: usize) -> f64 {
        self.sigma[iz * self.nx + ix]
    }

    /// The sponge profile, one value per cell.
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// `1 / dx^2` used by the five-point Laplacian.
    pub fn inv_dx2(&self) -> f64 {
        self.inv_dx2
    }

    /// Squared wave speed per cell.
    pub fn c2(&self) -> &[f64] {
        &self.c2
    }

    /// Squared wave speed of `background` (plus `perturbation` when asked).
    pub fn new(exp: &Experiment, perturbed: bool) -> Model {
        let c2 = exp
            .background
            .iter()
            .zip(&exp.perturbation)
            .map(|(&c, &p)| {
                let c = if perturbed { c + p } else { c };
                c * c
            })
            .collect();
        let mut sigma = vec![0.0; exp.nx * exp.nz];
        for iz in 0..exp.nz {
            for ix in 0..exp.nx {
                let d = ix
                    .min(exp.nx - 1 - ix)
                    .min(iz)
                    .min(exp.nz - 1 - iz) as f64;
                let t = 1.0 - d / exp.sponge_width;
                sigma[iz * exp.nx + ix] = exp.sponge_strength * t.max(0.0).powi(2);
            }
        }
        Model {
            nx: exp.nx,
            c2,
            sigma,
            inv_dx2: 1.0 / (exp.dx * exp.dx),
        }
    }
}

/// A Ricker pulse `(1 - 2a) exp(-a)` with `a = (pi f0 (t - t0))^2`.
pub fn ricker(exp: &Experiment, t: f64) -> f64 {
    let a = (std::f64::consts::PI * exp.f0 * (t - exp.t0)).powi(2);
    (1.0 - 2.0 * a) * (-a).exp()
}

pub struct Solver<'a> {
    exp: &'a Experiment,
    inv_dx2: f64,
}

impl<'a> Solver<'a> {
    pub fn new(exp: &'a Experiment) -> Solver<'a> {
        Solver {
            exp,
            inv_dx2: 1.0 / (exp.dx * exp.dx),
        }
    }

    /// Advance one shot.
    ///
    /// `traces` is `steps * receivers` row-major and receives `u^{n+1}` at
    /// every receiver. When `every` is set, `u^n` is also saved as a float32
    /// frame at steps `every, 2*every, ..., <= steps`, laid out `[frame][z][x]`.
    pub fn run(
        &self,
        shot: (f64, f64),
        model: &Model,
        every: Option<usize>,
        traces: &mut [f64],
    ) -> Vec<f32> {
        let exp = self.exp;
        let (nx, nz) = (exp.nx, exp.nz);
        let dt = exp.dt;
        let dt2 = dt * dt;

        // Unit-peak Gaussian footprint of the source, one value per cell.
        let (sx, sz) = shot;
        let footprint: Vec<f64> = (0..nz)
            .flat_map(|iz| {
                (0..nx).map(move |ix| {
                    let (dx, dz) = (ix as f64 - sx, iz as f64 - sz);
                    (-(dx * dx + dz * dz) / 2.0).exp()
                })
            })
            .collect();

        let mut frames: Vec<f32> = Vec::new();
        let mut previous = vec![0.0; nx * nz];
        let mut current = vec![0.0; nx * nz];
        let mut next = vec![0.0; nx * nz];

        for n in 0..exp.steps {
            let source = exp.amp * ricker(exp, n as f64 * dt);
            for iz in 1..nz - 1 {
                for ix in 1..nx - 1 {
                    let i = iz * nx + ix;
                    let lap = (current[i - 1] + current[i + 1] + current[i - nx] + current[i + nx]
                        - 4.0 * current[i])
                        * self.inv_dx2;
                    let q = source * footprint[i];
                    next[i] = (2.0 * current[i]
                        - (1.0 - model.sigma[i] * dt) * previous[i]
                        + dt2 * (model.c2[i] * lap + q))
                        / (1.0 + model.sigma[i] * dt);
                }
                // Zero boundary ring.
                next[iz * nx] = 0.0;
                next[iz * nx + nx - 1] = 0.0;
            }
            for ix in 0..nx {
                next[ix] = 0.0;
                next[(nz - 1) * nx + ix] = 0.0;
            }

            let step = n + 1;
            for (r, &(rx, rz)) in exp.receivers.iter().enumerate() {
                traces[n * exp.receivers.len() + r] = next[rz * nx + rx];
            }
            if let Some(every) = every {
                if step % every == 0 {
                    frames.extend(next.iter().map(|&v| v as f32));
                }
            }

            std::mem::swap(&mut previous, &mut current);
            std::mem::swap(&mut current, &mut next);
        }
        frames
    }
}
