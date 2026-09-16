use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ising", about = "Sample the two-dimensional Ising model")]
struct Args {
    #[arg(long)]
    update: String,
    #[arg(long = "l")]
    l: usize,
    #[arg(long = "t-from")]
    t_from: f64,
    #[arg(long = "t-to")]
    t_to: f64,
    #[arg(long = "t-step")]
    t_step: f64,
    #[arg(long)]
    discard: usize,
    #[arg(long)]
    measure: usize,
    #[arg(long, default_value_t = 0)]
    every: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Serialize)]
struct Run<'a> {
    #[serde(rename = "L")]
    l: usize,
    update: &'a str,
    t_grid: Vec<f64>,
    discard: usize,
    measure: usize,
    seed: u64,
    sample_every: usize,
    time_unit: &'static str,
}

#[derive(Serialize)]
struct Series {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "T")]
    t: f64,
    sweep: usize,
    #[serde(rename = "M")]
    m: f64,
    #[serde(rename = "E")]
    e: f64,
}

#[derive(Serialize)]
struct Frame {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "T")]
    t: f64,
    sweep: usize,
    m: f64,
    spins: Vec<i8>,
}

struct Ising {
    l: usize,
    spins: Vec<i8>,
}

impl Ising {
    fn new(l: usize) -> Self {
        Self {
            l,
            spins: vec![1; l * l],
        }
    }

    fn index(&self, row: usize, col: usize) -> usize {
        (row % self.l) * self.l + (col % self.l)
    }

    fn delta_energy(&self, index: usize) -> f64 {
        let row = index / self.l;
        let col = index % self.l;
        let neighbours = self.spins[self.index(row + 1, col)]
            + self.spins[self.index(row + self.l - 1, col)]
            + self.spins[self.index(row, col + 1)]
            + self.spins[self.index(row, col + self.l - 1)];
        2.0 * f64::from(self.spins[index]) * f64::from(neighbours)
    }

    fn sweep(&mut self, temperature: f64, rng: &mut StdRng) -> usize {
        let mut accepted = 0;
        for _ in 0..self.l * self.l {
            let index = rng.gen_range(0..self.spins.len());
            let delta = self.delta_energy(index);
            if delta <= 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
                self.spins[index] = -self.spins[index];
                accepted += 1;
            }
        }
        accepted
    }

    fn magnetization(&self) -> f64 {
        self.spins.iter().map(|&spin| f64::from(spin)).sum::<f64>() / self.spins.len() as f64
    }

    fn energy(&self) -> f64 {
        let mut energy = 0.0;
        for row in 0..self.l {
            for col in 0..self.l {
                let index = self.index(row, col);
                energy -= f64::from(self.spins[index])
                    * f64::from(
                        self.spins[self.index(row + 1, col)] + self.spins[self.index(row, col + 1)],
                    );
            }
        }
        energy / self.spins.len() as f64
    }
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn temperature_grid(from: f64, to: f64, step: f64) -> Vec<f64> {
    let mut grid = Vec::new();
    let mut temperature = from;
    while temperature <= to + step.abs() * 1e-10 {
        grid.push(round6(temperature));
        temperature += step;
    }
    grid
}

fn run(args: Args) -> Result<(), String> {
    if args.update != "metropolis" {
        return Err("only metropolis is implemented".into());
    }
    if args.l < 2 || args.t_step <= 0.0 || args.t_from <= 0.0 || args.t_to < args.t_from {
        return Err("invalid lattice or temperature range".into());
    }
    let t_grid = temperature_grid(args.t_from, args.t_to, args.t_step);
    if t_grid.is_empty() {
        return Err("temperature grid is empty".into());
    }
    fs::create_dir_all(&args.out).map_err(|e| format!("create output folder: {e}"))?;
    let run_file =
        File::create(args.out.join("run.json")).map_err(|e| format!("open run.json: {e}"))?;
    serde_json::to_writer_pretty(
        run_file,
        &Run {
            l: args.l,
            update: &args.update,
            t_grid: t_grid.clone(),
            discard: args.discard,
            measure: args.measure,
            seed: args.seed,
            sample_every: args.every,
            time_unit: "sweep",
        },
    )
    .map_err(|e| format!("write run.json: {e}"))?;
    let mut series = BufWriter::new(
        File::create(args.out.join("series.jsonl"))
            .map_err(|e| format!("open series.jsonl: {e}"))?,
    );
    let mut spins_file = if args.every > 0 {
        Some(BufWriter::new(
            File::create(args.out.join("spins.jsonl"))
                .map_err(|e| format!("open spins.jsonl: {e}"))?,
        ))
    } else {
        None
    };
    println!("T\tmean_abs_M\tacceptance_rate");
    let mut model = Ising::new(args.l);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut cumulative_sweep = 0;
    for &temperature in &t_grid {
        let mut accepted = 0usize;
        for _ in 0..args.discard {
            accepted += model.sweep(temperature, &mut rng);
            cumulative_sweep += 1;
        }
        let mut abs_m_sum = 0.0;
        for sweep in 1..=args.measure {
            accepted += model.sweep(temperature, &mut rng);
            cumulative_sweep += 1;
            let m = round6(model.magnetization());
            let e = round6(model.energy());
            abs_m_sum += m.abs();
            serde_json::to_writer(
                &mut series,
                &Series {
                    l: args.l,
                    t: temperature,
                    sweep,
                    m,
                    e,
                },
            )
            .map_err(|e| format!("write series: {e}"))?;
            series
                .write_all(b"\n")
                .map_err(|e| format!("write series newline: {e}"))?;
            if let Some(file) = spins_file.as_mut() {
                if sweep % args.every == 0 {
                    serde_json::to_writer(
                        &mut *file,
                        &Frame {
                            l: args.l,
                            t: temperature,
                            sweep: cumulative_sweep,
                            m,
                            spins: model.spins.clone(),
                        },
                    )
                    .map_err(|e| format!("write spins: {e}"))?;
                    file.write_all(b"\n")
                        .map_err(|e| format!("write spins newline: {e}"))?;
                }
            }
        }
        let proposals = (args.discard + args.measure) * args.l * args.l;
        let acceptance = if proposals == 0 {
            0.0
        } else {
            accepted as f64 / proposals as f64
        };
        println!(
            "{temperature:.6}\t{:.6}\t{acceptance:.6}",
            if args.measure == 0 {
                0.0
            } else {
                abs_m_sum / args.measure as f64
            }
        );
    }
    series.flush().map_err(|e| format!("flush series: {e}"))?;
    if let Some(mut file) = spins_file {
        file.flush().map_err(|e| format!("flush spins: {e}"))?;
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(error) = run(args) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}
