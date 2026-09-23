//! Time-step convergence sweep for the same periodic Gaussian pulse used by
//! `accuracy_run`. Confirms the order of each scheme and shows that the
//! centred-difference error is limited by spatial phase error, not by dt.
//!
//! Usage: convergence

use advection::{
    gaussian, CentredDifferenceRhs, ForwardEuler, FourierRhs, Integrator, PeriodicGrid,
    RungeKutta4,
};
use std::f64::consts::PI;

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.002;
const T_END: f64 = 2.0 * PI;

fn max_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn sweep(
    label: &str,
    dts: &[f64],
    run: impl Fn(&[f64], f64) -> Vec<f64>,
    exact_at: impl Fn(f64) -> Vec<f64>,
) {
    let initial = gaussian(&PeriodicGrid::new(N), 0.25, PI / 2.0);
    println!("{label}");
    let mut previous: Option<(f64, f64)> = None;
    for &dt in dts {
        let steps = (T_END / dt).round() as usize;
        let t_final = steps as f64 * dt;
        let error = max_error(&run(&initial, dt), &exact_at(t_final));
        let order = match previous {
            Some((prev_dt, prev_err)) => {
                let p = (prev_err / error).ln() / (prev_dt / dt).ln();
                format!("   order {p:.2}")
            }
            None => String::new(),
        };
        println!("  dt = {dt:<8} err = {error:.4e}{order}");
        previous = Some((dt, error));
    }
}

fn main() {
    let grid = PeriodicGrid::new(N);
    let initial = gaussian(&grid, 0.25, PI / 2.0);
    let fourier = FourierRhs::new(grid, C, NU);
    let centred = CentredDifferenceRhs::new(grid, C, NU);

    sweep(
        "RK4 Fourier",
        &[0.04, 0.02, 0.01, 0.005],
        |s, dt| RungeKutta4.advance(s, dt, (T_END / dt).round() as usize, |v| fourier.rate(v)),
        |t| fourier.exact(&initial, t),
    );
    sweep(
        "Forward Euler Fourier",
        &[0.01, 0.005, 0.0025, 0.00125],
        |s, dt| ForwardEuler.advance(s, dt, (T_END / dt).round() as usize, |v| fourier.rate(v)),
        |t| fourier.exact(&initial, t),
    );
    sweep(
        "RK4 centred",
        &[0.02, 0.01, 0.005, 0.0025],
        |s, dt| RungeKutta4.advance(s, dt, (T_END / dt).round() as usize, |v| centred.rate(v)),
        |t| fourier.exact(&initial, t),
    );
}
