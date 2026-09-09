use clap::{Args, Parser, Subcommand};
use md::fluid::{check_artifacts, run_fluid, write_artifacts, FluidConfig, MdError};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "md",
    about = "Two-dimensional Lennard-Jones molecular dynamics"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Run(RunArgs),
    Check(CheckArgs),
    Video(VideoArgs),
}

#[derive(Args)]
struct RunArgs {
    #[arg(long, default_value_t = 100)]
    n: usize,
    #[arg(long, default_value_t = 0.8)]
    rho: f64,
    #[arg(long, default_value_t = 0.5)]
    temperature: f64,
    #[arg(long, default_value_t = 0.01)]
    dt: f64,
    #[arg(long, default_value_t = 2000)]
    eq_steps: usize,
    #[arg(long, default_value_t = 10000)]
    steps: usize,
    #[arg(long, default_value_t = 50)]
    sample_every: usize,
    #[arg(long, default_value_t = 2026)]
    seed: u64,
    #[arg(long, default_value = "artifacts")]
    out: PathBuf,
}

#[derive(Args)]
struct CheckArgs {
    #[arg(long, default_value = "artifacts")]
    input: PathBuf,
}

#[derive(Args)]
struct VideoArgs {
    #[arg(long, default_value = "artifacts")]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 20)]
    fps: u32,
    #[arg(long, default_value_t = 48)]
    rdf_bins: usize,
    #[arg(long, default_value_t = 20)]
    rdf_window: usize,
}

fn main() -> ExitCode {
    match execute(Cli::parse()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn execute(cli: Cli) -> Result<ExitCode, MdError> {
    match cli.command {
        None => {
            println!("Hello, world!");
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Run(args)) => {
            let config = FluidConfig {
                n: args.n,
                rho: args.rho,
                temperature: args.temperature,
                dt: args.dt,
                eq_steps: args.eq_steps,
                steps: args.steps,
                sample_every: args.sample_every,
                seed: args.seed,
                ..FluidConfig::default()
            };
            let simulation = run_fluid(&config)?;
            write_artifacts(&args.out, &simulation)?;
            println!(
                "wrote {} frames to {}",
                simulation.frames.len(),
                args.out.display()
            );
            Ok(ExitCode::SUCCESS)
        }
        Some(Command::Check(args)) => {
            let report = check_artifacts(&args.input)?;
            println!(
                "Stored energy integrity : {} ({:.3e})",
                pass(report.energy_integrity),
                report.maximum_energy_mismatch
            );
            println!(
                "Momentum conservation   : {} ({:.3e})",
                pass(report.momentum_conservation),
                report.maximum_momentum_per_particle
            );
            println!(
                "Production energy drift : {} ({:.3e})",
                pass(report.energy_drift),
                report.rolling_energy_drift
            );
            println!(
                "max instantaneous error : {:.3e}",
                report.maximum_relative_energy_error
            );
            println!("mean temperature        : {:.6}", report.mean_temperature);
            println!(
                "minimum pair distance   : {:.6}",
                report.minimum_pair_distance
            );
            Ok(if report.passed() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        Some(Command::Video(args)) => {
            let output = args.output.unwrap_or_else(|| args.input.join("md.mp4"));
            md::fluid::encode_video(
                &args.input,
                &output,
                args.fps,
                args.rdf_bins,
                args.rdf_window,
            )?;
            println!("wrote {}", output.display());
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn pass(value: bool) -> &'static str {
    if value {
        "PASS"
    } else {
        "FAIL"
    }
}
