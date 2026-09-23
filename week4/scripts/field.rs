//! `field` writes a two-dimensional velocity field to stdout as one JSON
//! object, for `fluid` to integrate.
//!
//! Cases:
//!   taylor-green --n N [--nu NU] [--t T]
//!     u = cos(x) sin(y) exp(-2 nu t), v = -sin(x) cos(y) exp(-2 nu t)
//!   random --n N --seed S --k-min A --k-max B
//!     vorticity from equal-amplitude cosine modes, A <= |k| <= B, E(0) = 0.5
//!
//! Arrays are row-major with row = y index and column = x index.

use advection::vorticity::{Grid2d, Spectral};
use clap::{Args, Parser, Subcommand};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "field", about = "Write a velocity field as JSON to stdout")]
struct Cli {
    #[command(subcommand)]
    case: Case,
}

#[derive(Subcommand)]
enum Case {
    /// u = cos(x) sin(y) exp(-2 nu t), v = -sin(x) cos(y) exp(-2 nu t)
    #[command(name = "taylor-green")]
    TaylorGreen(TaylorGreenArgs),
    /// vorticity from equal-amplitude modes, k-min <= |k| <= k-max, E(0) = 0.5
    Random(RandomArgs),
}

#[derive(Args)]
struct TaylorGreenArgs {
    /// grid points per side
    #[arg(long)]
    n: usize,
    /// time of the exact solution; default 0, the initial field
    #[arg(long, default_value_t = 0.0)]
    t: f64,
    /// viscosity of the decay; required if t > 0
    #[arg(long)]
    nu: Option<f64>,
}

#[derive(Args)]
struct RandomArgs {
    /// grid points per side
    #[arg(long)]
    n: usize,
    /// seed of the phases
    #[arg(long)]
    seed: u64,
    /// lowest |k| kept, inclusive
    #[arg(long = "k-min")]
    k_min: f64,
    /// highest |k| kept, inclusive
    #[arg(long = "k-max")]
    k_max: f64,
}

fn main() {
    let cli = Cli::parse();
    let json = match cli.case {
        Case::TaylorGreen(args) => taylor_green(args),
        Case::Random(args) => random(args),
    };
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(json.as_bytes()).expect("write stdout");
    handle.write_all(b"\n").expect("write newline");
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}

fn taylor_green(args: TaylorGreenArgs) -> String {
    if args.n < 6 {
        fail("n must be at least 6");
    }
    let grid = Grid2d::new(args.n);
    let nu = args.nu.unwrap_or(0.0);
    if args.t > 0.0 && args.nu.is_none() {
        fail("--nu is required when --t > 0");
    }

    let decay = (-2.0 * nu * args.t).exp();
    let mut u = vec![0.0; grid.len()];
    let mut v = vec![0.0; grid.len()];
    for iy in 0..grid.n() {
        for ix in 0..grid.n() {
            let (x, y) = (grid.x(ix), grid.y(iy));
            let idx = grid.flat(iy, ix);
            u[idx] = x.cos() * y.sin() * decay;
            v[idx] = -x.sin() * y.cos() * decay;
        }
    }

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"case\": \"taylor-green\",\n");
    json.push_str(&format!("  \"n\": {},\n", args.n));
    json.push_str("  \"seed\": null,\n");
    json.push_str("  \"k_band\": [1, 1],\n");
    push_array(&mut json, "u", &u);
    json.push_str(",\n");
    push_array(&mut json, "v", &v);
    json.push_str("\n}\n");
    json
}

fn random(args: RandomArgs) -> String {
    if args.n < 6 {
        fail("n must be at least 6");
    }
    if args.k_min <= 0.0 {
        fail("k-min must be positive");
    }
    if args.k_min > args.k_max {
        fail("k-min must not exceed k-max");
    }
    let grid = Grid2d::new(args.n);
    let kmax = grid.dealias_kmax() as f64;
    if args.k_max > kmax {
        fail(&format!(
            "k-max {} exceeds the two-thirds limit {kmax}",
            args.k_max
        ));
    }

    let lo = args.k_min.ceil() as i64;
    let hi = args.k_max.ceil() as i64;

    let mut modes: Vec<(i64, i64)> = Vec::new();
    for ky in -hi..=hi {
        for kx in -hi..=hi {
            let radius = ((kx * kx + ky * ky) as f64).sqrt();
            if radius >= args.k_min - 1e-12 && radius <= args.k_max + 1e-12 {
                modes.push((kx, ky));
            }
        }
    }
    if modes.is_empty() {
        fail("no modes in the requested band");
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let phases: Vec<f64> = (0..modes.len())
        .map(|_| rng.gen_range(0.0..std::f64::consts::TAU))
        .collect();

    let mut omega = vec![0.0; grid.len()];
    for (mode, &phase) in modes.iter().zip(&phases) {
        let (kx, ky) = *mode;
        for iy in 0..grid.n() {
            for ix in 0..grid.n() {
                let angle = kx as f64 * grid.x(ix) + ky as f64 * grid.y(iy) + phase;
                omega[grid.flat(iy, ix)] += angle.cos();
            }
        }
    }

    let spectral = Spectral::new(Grid2d::new(args.n), 0.0);
    let omega_hat = spectral.forward_real(&omega);
    let (mut u, mut v) = spectral.velocity(&omega_hat);
    let energy = spectral.energy(&u, &v);
    if energy <= 0.0 {
        fail("generated field has no energy");
    }
    let scale = (0.5 / energy).sqrt();
    for value in u.iter_mut() {
        *value *= scale;
    }
    for value in v.iter_mut() {
        *value *= scale;
    }

    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"case\": \"random\",\n");
    json.push_str(&format!("  \"n\": {},\n", args.n));
    json.push_str(&format!("  \"seed\": {},\n", args.seed));
    json.push_str(&format!("  \"k_band\": [{}, {}],\n", lo, hi));
    push_array(&mut json, "u", &u);
    json.push_str(",\n");
    push_array(&mut json, "v", &v);
    json.push_str("\n}\n");
    json
}

fn push_array(out: &mut String, name: &str, values: &[f64]) {
    out.push_str(&format!("  \"{name}\": ["));
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{value:.6}"));
    }
    out.push(']');
}
