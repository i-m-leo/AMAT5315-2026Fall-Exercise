//! `fluid` integrates the two-dimensional incompressible vorticity equation on
//! a periodic `[0, 2pi)^2` grid, reading the field that `field` writes to
//! stdout and recording snapshots.
//!
//! Usage:
//!   field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 \
//!       --t-end 1 --every 0.1 --out artifacts/taylor-green
//!
//! The initial vorticity is the spectral curl of the piped velocity,
//! `omega_hat = i kx v_hat - i ky u_hat`. Integration uses the library
//! integrators on an interleaved real/imaginary spectral state, with the
//! two-thirds rule applied to the vorticity and to every product.

use advection::integrators::Scheme;
use advection::vorticity::{Grid2d, Spectral};
use clap::Parser;
use serde::Serialize;
use serde_json::json;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "fluid", about = "Integrate a 2D vorticity field read from stdin")]
struct Args {
    /// time integrator: euler, rk2, or rk4
    #[arg(long)]
    method: String,
    /// kinematic viscosity
    #[arg(long)]
    nu: f64,
    /// time step
    #[arg(long)]
    dt: f64,
    /// final integration time
    #[arg(long = "t-end")]
    t_end: f64,
    /// time between snapshots
    #[arg(long)]
    every: f64,
    /// output folder
    #[arg(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct Run {
    case: String,
    n: usize,
    seed: Option<u64>,
    k_band: Option<Vec<f64>>,
    method: String,
    nu: f64,
    dt: f64,
    t_end: f64,
    snapshot_every: usize,
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let scheme = Scheme::parse_cli(&args.method)
        .ok_or_else(|| format!("unknown method '{}', expected euler, rk2, or rk4", args.method))?;
    if args.dt <= 0.0 {
        return Err("dt must be positive".to_string());
    }
    if args.every <= 0.0 {
        return Err("every must be positive".to_string());
    }
    if args.every < args.dt {
        return Err("every must be at least dt".to_string());
    }

    let input = read_stdin();
    let parsed: serde_json::Value =
        serde_json::from_str(&input).map_err(|e| format!("parse stdin JSON: {e}"))?;
    let case = parsed
        .get("case")
        .and_then(|v| v.as_str())
        .ok_or("stdin JSON is missing 'case'")?
        .to_string();
    let n = parsed
        .get("n")
        .and_then(|v| v.as_u64())
        .ok_or("stdin JSON is missing 'n'")? as usize;
    let seed = parsed.get("seed").and_then(|v| v.as_u64());
    let k_band = parsed.get("k_band").and_then(|v| {
        v.as_array().map(|values| {
            values
                .iter()
                .map(|value| value.as_f64().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        })
    });

    if n < 6 {
        return Err("n must be at least 6".to_string());
    }

    let grid = Grid2d::new(n);
    let u = read_array(&parsed, "u", grid.len())?;
    let v = read_array(&parsed, "v", grid.len())?;

    let spectral = Spectral::new(Grid2d::new(n), args.nu);
    let omega_hat = spectral.vorticity_hat(&u, &v);
    let mut state = spectral.state_from_hat(&omega_hat);

    let steps = (args.t_end / args.dt).round() as usize;
    let snapshot_every = (args.every / args.dt).round() as usize;
    let snapshot_every = snapshot_every.max(1);

    std::fs::create_dir_all(&args.out).map_err(|e| format!("create out dir: {e}"))?;
    let run_record = Run {
        case,
        n,
        seed,
        k_band,
        method: scheme.cli_name().to_string(),
        nu: args.nu,
        dt: args.dt,
        t_end: args.t_end,
        snapshot_every,
    };
    let run_json = serde_json::to_string_pretty(&run_record).map_err(|e| e.to_string())?;
    std::fs::write(args.out.join("run.json"), run_json + "\n")
        .map_err(|e| format!("write run.json: {e}"))?;

    let fields_file = File::create(args.out.join("fields.jsonl"))
        .map_err(|e| format!("open fields.jsonl: {e}"))?;
    let mut fields = BufWriter::new(fields_file);

    let stdout = io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "t\tE\tZ").map_err(|e| e.to_string())?;

    let mut failed = false;
    let mut step = 0usize;
    let mut t = 0.0_f64;
    loop {
        let (u_now, v_now) = spectral.velocity(&spectral.hat_from_state(&state));
        let omega = spectral.physical_omega(&spectral.hat_from_state(&state));
        let energy = spectral.energy(&u_now, &v_now);
        let enstrophy = spectral.enstrophy(&omega);

        writeln!(out, "{t:.6}\t{energy:.6e}\t{enstrophy:.6e}").map_err(|e| e.to_string())?;

        if !energy.is_finite() || !enstrophy.is_finite() {
            failed = true;
            break;
        }

        if step % snapshot_every == 0 {
            write_frame(&mut fields, t, step, &u_now, &v_now, &omega)?;
        }

        if step == steps {
            break;
        }

        state = scheme.step(&state, args.dt, |s| spectral.rhs(s));
        step += 1;
        t = step as f64 * args.dt;
    }

    fields.flush().map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())?;

    if failed {
        std::process::exit(1);
    }
    Ok(())
}

fn write_frame(
    writer: &mut BufWriter<File>,
    t: f64,
    step: usize,
    u: &[f64],
    v: &[f64],
    omega: &[f64],
) -> Result<(), String> {
    let frame = json!({
        "t": round6(t),
        "step": step,
        "u": u.iter().map(|&value| round6(value)).collect::<Vec<f64>>(),
        "v": v.iter().map(|&value| round6(value)).collect::<Vec<f64>>(),
        "omega": omega.iter().map(|&value| round6(value)).collect::<Vec<f64>>(),
    });
    writeln!(writer, "{frame}").map_err(|e| format!("write fields.jsonl: {e}"))
}

fn read_array(parsed: &serde_json::Value, key: &str, expected: usize) -> Result<Vec<f64>, String> {
    let values = parsed
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("stdin JSON is missing '{key}'"))?;
    if values.len() != expected {
        return Err(format!(
            "'{key}' has {} entries, expected {expected}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| value.as_f64().ok_or_else(|| format!("'{key}' has a non-numeric entry")))
        .collect()
}

fn read_stdin() -> String {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("read stdin");
    input
}
