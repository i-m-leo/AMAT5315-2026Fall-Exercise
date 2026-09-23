//! One-step time integrators sharing a single trait.
//!
//! The trait works on a state vector and a rate function that maps a state to
//! its time derivative. The rate does not depend on time, which is enough for
//! the advection-diffusion equation.

pub trait Integrator {
    fn name(&self) -> &'static str;

    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>;

    fn advance<F>(&self, state: &[f64], dt: f64, steps: usize, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut current = state.to_vec();
        for _ in 0..steps {
            current = self.step(&current, dt, &rate);
        }
        current
    }
}

fn axpy(base: &[f64], alpha: f64, slope: &[f64]) -> Vec<f64> {
    base.iter()
        .zip(slope)
        .map(|(b, s)| b + alpha * s)
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn name(&self) -> &'static str {
        "forward-euler"
    }

    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        axpy(state, dt, &rate(state))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Midpoint;

impl Integrator for Midpoint {
    fn name(&self) -> &'static str {
        "explicit-midpoint"
    }

    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let predictor = axpy(state, 0.5 * dt, &k1);
        let k2 = rate(&predictor);
        axpy(state, dt, &k2)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RungeKutta4;

impl Integrator for RungeKutta4 {
    fn name(&self) -> &'static str {
        "classical-rk4"
    }

    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rate(state);
        let k2 = rate(&axpy(state, 0.5 * dt, &k1));
        let k3 = rate(&axpy(state, 0.5 * dt, &k2));
        let k4 = rate(&axpy(state, dt, &k3));
        state
            .iter()
            .zip(&k1)
            .zip(&k2)
            .zip(&k3)
            .zip(&k4)
            .map(|((((u, a), b), c), d)| u + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    ForwardEuler,
    Midpoint,
    RungeKutta4,
}

impl Scheme {
    pub fn all() -> [Scheme; 3] {
        [Scheme::ForwardEuler, Scheme::Midpoint, Scheme::RungeKutta4]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Scheme::ForwardEuler => ForwardEuler.name(),
            Scheme::Midpoint => Midpoint.name(),
            Scheme::RungeKutta4 => RungeKutta4.name(),
        }
    }

    pub fn advance<F>(&self, state: &[f64], dt: f64, steps: usize, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        match self {
            Scheme::ForwardEuler => ForwardEuler.advance(state, dt, steps, rate),
            Scheme::Midpoint => Midpoint.advance(state, dt, steps, rate),
            Scheme::RungeKutta4 => RungeKutta4.advance(state, dt, steps, rate),
        }
    }
}
