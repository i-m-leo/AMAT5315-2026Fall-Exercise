//! Record u(x, t) for a periodic Gaussian pulse under the library's RK4.
//!
//! Usage: pulse-field dt [t_end]
//!
//! Initial state: a Gaussian of standard deviation `sigma = 0.35` centred at
//! `x0 = pi/2`, wrapped with the minimum-image distance so it is periodic.
//! Integrated with `FourierRhs` at `nu = 0.05`, `c = 1` on `n = 64` points.
//! Output: a `#` header line, then one row per time level as
//! `t u_0 u_1 ... u_{n-1}`.

use advection::{FourierRhs, Integrator, PeriodicGrid, RungeKutta4};
use std::f64::consts::PI;

fn gaussian(grid: &PeriodicGrid, sigma: f64, x0: f64) -> Vec<f64> {
    let length = grid.length();
    grid.points()
        .into_iter()
        .map(|x| {
            let raw = (x - x0).abs();
            let d = raw.min(length - raw);
            (-0.5 * (d / sigma).powi(2)).exp()
        })
        .collect()
}

fn print_row(t: f64, u: &[f64]) {
    print!("{:.10e}", t);
    for value in u {
        print!(" {:.10e}", value);
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dt: f64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(0.045);
    let t_end: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(6.0);

    let n = 64;
    let sigma = 0.35;
    let x0 = PI / 2.0;
    let nu = 0.05;
    let c = 1.0;

    let grid = PeriodicGrid::new(n);
    let rhs = FourierRhs::new(grid, c, nu);
    let steps = (t_end / dt).round() as usize;
    let actual_t_end = steps as f64 * dt;

    let mut u = gaussian(&grid, sigma, x0);
    let peak = |v: &[f64]| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));

    println!(
        "# n={} nu={} c={} sigma={} x0={:.10} dt={:.10} steps={} t_end={:.10}",
        n, nu, c, sigma, x0, dt, steps, actual_t_end
    );
    print_row(0.0, &u);
    for step in 1..=steps {
        u = RungeKutta4.step(&u, dt, |v| rhs.rate(v));
        print_row(step as f64 * dt, &u);
    }
    eprintln!("peak |u| at t={actual_t_end:.6}: {:.6e}", peak(&u));
}
