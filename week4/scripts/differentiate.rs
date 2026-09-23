//! Compare the solver's spectral derivatives of g(x, y) = sin(3x) cos(2y)
//! against the analytic derivatives, and against second-order centred finite
//! differences on n = 32 and n = 64.
//!
//! Analytic:
//!   g_x  = 3 cos(3x) cos(2y)
//!   g_xx = -9 sin(3x) cos(2y)
//!   g_xy = -6 cos(3x) sin(2y)
//!   L g  = -13 sin(3x) cos(2y)
//!
//! Usage: differentiate

use advection::{centred_derivative, Derivative, Grid2d, Spectral};

const OPS: [(Derivative, &str); 4] = [
    (Derivative::X, "dx"),
    (Derivative::XX, "dxx"),
    (Derivative::XY, "dxdy"),
    (Derivative::Laplacian, "lap"),
];

fn field(grid: &Grid2d) -> Vec<f64> {
    let mut values = vec![0.0; grid.len()];
    for iy in 0..grid.n() {
        for ix in 0..grid.n() {
            let (x, y) = (grid.x(ix), grid.y(iy));
            values[grid.flat(iy, ix)] = (3.0 * x).sin() * (2.0 * y).cos();
        }
    }
    values
}

fn analytic(grid: &Grid2d, op: Derivative) -> Vec<f64> {
    let mut values = vec![0.0; grid.len()];
    for iy in 0..grid.n() {
        for ix in 0..grid.n() {
            let (x, y) = (grid.x(ix), grid.y(iy));
            let (s3x, c3x) = ((3.0 * x).sin(), (3.0 * x).cos());
            let (s2y, c2y) = ((2.0 * y).sin(), (2.0 * y).cos());
            values[grid.flat(iy, ix)] = match op {
                Derivative::X => 3.0 * c3x * c2y,
                Derivative::XX => -9.0 * s3x * c2y,
                Derivative::XY => -6.0 * c3x * s2y,
                Derivative::Laplacian => -13.0 * s3x * c2y,
            };
        }
    }
    values
}

fn max_abs_error(computed: &[f64], exact: &[f64]) -> f64 {
    computed
        .iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let n = 32;
    let grid = Grid2d::new(n);
    let values = field(&grid);
    let spectral = Spectral::new(grid, 0.0);

    println!("g(x, y) = sin(3x) cos(2y) on [0, 2pi)^2; maximum absolute error");
    println!();
    println!(
        "{:<6} {:>14} {:>14} {:>14} {:>14}",
        "op", "spectral n=32", "centred n=32", "centred n=64", "ratio 32/64"
    );

    let grid64 = Grid2d::new(64);
    let values64 = field(&grid64);

    for (op, name) in OPS {
        let exact = analytic(&grid, op);
        let exact64 = analytic(&grid64, op);

        let spectral_error = max_abs_error(&spectral.differentiate(&values, op), &exact);
        let centred_error = max_abs_error(&centred_derivative(&values, n, op), &exact);
        let centred_error64 = max_abs_error(&centred_derivative(&values64, 64, op), &exact64);
        let ratio = centred_error / centred_error64;

        println!(
            "{:<6} {:>14.3e} {:>14.3e} {:>14.3e} {:>14.3e}",
            name, spectral_error, centred_error, centred_error64, ratio
        );
    }
}
