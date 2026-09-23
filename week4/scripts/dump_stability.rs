//! Dump the measured per-step growth factor of each integrator on a grid of
//! complex `z = lambda h`.
//!
//! For each `z` we advance `y' = lambda y` one step of size `h = 1` from
//! `y = 1 + 0i`. The integrator is fed a two-element real state `(Re y, Im y)`
//! and the rate `lambda y`. `|y_1|` is the growth factor per step.
//!
//! Usage: dump_stability zmax ngrid
//! Real/imaginary box is `[-zmax, zmax]` in both directions, `ngrid` points per
//! axis. Output lines: `name z_re z_im |growth|`.

use advection::Scheme;

fn rate(lambda: (f64, f64), state: &[f64]) -> Vec<f64> {
    let (a, b) = (state[0], state[1]);
    vec![lambda.0 * a - lambda.1 * b, lambda.1 * a + lambda.0 * b]
}

fn growth(scheme: Scheme, lambda: (f64, f64)) -> f64 {
    let out = scheme.advance(&[1.0, 0.0], 1.0, 1, |s| rate(lambda, s));
    (out[0] * out[0] + out[1] * out[1]).sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let zmax: f64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(3.5);
    let ngrid: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(321);

    for scheme in Scheme::all() {
        println!("# {}", scheme.name());
        for i in 0..ngrid {
            let re = -zmax + 2.0 * zmax * i as f64 / (ngrid - 1) as f64;
            for j in 0..ngrid {
                let im = -zmax + 2.0 * zmax * j as f64 / (ngrid - 1) as f64;
                println!(
                    "{} {:.10e} {:.10e} {:.10e}",
                    scheme.name(),
                    re,
                    im,
                    growth(scheme, (re, im))
                );
            }
        }
    }
}
