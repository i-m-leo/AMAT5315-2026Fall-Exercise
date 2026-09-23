//! Time-step convergence of four one-step methods on the periodic Gaussian
//! pulse, using the library's Fourier derivatives.
//!
//! Pulse: sigma = 0.35 centred at x = pi/2, wrapped periodically.
//! Grid: n = 64, c = 1, nu = 0.05, integrated to t = 1.
//! Methods: forward Euler, explicit midpoint, classical RK4, and an RK4 whose
//! four stage weights are equal (b = 1/4 each).
//!
//! Usage: method-convergence
//! Writes `evidence/method-convergence.txt` as `method dt max_error` lines.

use advection::{gaussian, ForwardEuler, FourierRhs, Integrator, Midpoint, PeriodicGrid, RungeKutta4};
use std::f64::consts::PI;

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.05;
const SIGMA: f64 = 0.35;
const X0: f64 = PI / 2.0;
const T_END: f64 = 1.0;
const DTS: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];

/// Classical RK4 stages with equal weights b_i = 1/4 instead of
/// (1/6, 1/3, 1/3, 1/6).
#[derive(Clone, Copy, Debug, Default)]
struct EqualWeightRk4;

impl Integrator for EqualWeightRk4 {
    fn name(&self) -> &'static str {
        "rk4-equal-weights"
    }

    fn step<F>(&self, state: &[f64], dt: f64, rate: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let axpy = |base: &[f64], alpha: f64, slope: &[f64]| -> Vec<f64> {
            base.iter()
                .zip(slope)
                .map(|(b, s)| b + alpha * s)
                .collect()
        };
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
            .map(|((((u, a), b), c), d)| u + 0.25 * dt * (a + b + c + d))
            .collect()
    }
}

fn max_error(computed: &[f64], exact: &[f64]) -> f64 {
    computed
        .iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "evidence".to_string());
    std::fs::create_dir_all(&out_dir).expect("create output directory");

    let grid = PeriodicGrid::new(N);
    let initial = gaussian(&grid, SIGMA, X0);
    let fourier = FourierRhs::new(grid, C, NU);

    let methods: [(&str, Box<dyn Fn(&[f64], f64, usize) -> Vec<f64>>); 4] = [
        (
            "forward-euler",
            Box::new(|s, dt, steps| ForwardEuler.advance(s, dt, steps, |v| fourier.rate(v))),
        ),
        (
            "explicit-midpoint",
            Box::new(|s, dt, steps| Midpoint.advance(s, dt, steps, |v| fourier.rate(v))),
        ),
        (
            "classical-rk4",
            Box::new(|s, dt, steps| RungeKutta4.advance(s, dt, steps, |v| fourier.rate(v))),
        ),
        (
            "rk4-equal-weights",
            Box::new(|s, dt, steps| EqualWeightRk4.advance(s, dt, steps, |v| fourier.rate(v))),
        ),
    ];

    let mut out = String::from("method dt max_error\n");
    for (name, run) in &methods {
        for &dt in &DTS {
            let steps = (T_END / dt).round() as usize;
            let t_final = steps as f64 * dt;
            let values = run(&initial, dt, steps);
            let exact = fourier.exact(&initial, t_final);
            out.push_str(&format!("{name} {dt:.10e} {:.10e}\n", max_error(&values, &exact)));
        }
    }

    let path = std::path::Path::new(&out_dir).join("method-convergence.txt");
    std::fs::write(&path, &out).expect("write convergence table");
    print!("{out}");
}
