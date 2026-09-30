//! `seismic` runs forward acoustic modelling on a week5 experiment file.
//!
//! ```text
//! seismic --experiment inputs/reflector.json --mode forward --every 3 \
//!         --out artifacts/forward
//! ```

use clap::{Parser, ValueEnum};
use seismic::npy;
use seismic::output;
use seismic::{Experiment, Model, Solver};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Forward,
    Born,
    Adjoint,
}

#[derive(Parser)]
#[command(name = "seismic", about = "Simulate acoustic waves and their derivatives")]
struct Cli {
    /// JSON file with the model and the survey
    #[arg(long)]
    experiment: PathBuf,
    /// forward: pressure traces; born: Born scattered data; adjoint: migrated image
    #[arg(long, value_enum)]
    mode: Mode,
    /// output folder
    #[arg(long)]
    out: PathBuf,
    /// forward and adjoint only: frame interval in steps from step 0; default off
    #[arg(long)]
    every: Option<usize>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Mode::Born | Mode::Adjoint = cli.mode {
        eprintln!("mode {:?} is not implemented yet", mode_name(cli.mode));
        return ExitCode::from(1);
    }
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("seismic: {message}");
            ExitCode::from(1)
        }
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Forward => "forward",
        Mode::Born => "born",
        Mode::Adjoint => "adjoint",
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    if let Some(0) = cli.every {
        return Err("--every must be at least 1".into());
    }
    let exp = Experiment::load(&cli.experiment)?;
    let solver = Solver::new(&exp);
    let n_receivers = exp.receivers.len();
    let n_cells = exp.nx * exp.nz;
    let background = Model::new(&exp, false);

    let mut traces = vec![0.0; exp.shots.len() * exp.steps * n_receivers];
    let mut frames: Option<Vec<f32>> = None;
    for (s, &shot) in exp.shots.iter().enumerate() {
        let start = s * exp.steps * n_receivers;
        let shot_frames = solver.run(
            shot,
            &background,
            cli.every,
            &mut traces[start..start + exp.steps * n_receivers],
        );
        if s == 0 {
            frames = Some(shot_frames);
        }
    }

    std::fs::create_dir_all(&cli.out)
        .map_err(|e| format!("cannot create {}: {e}", cli.out.display()))?;
    npy::write_f64(
        &cli.out.join("traces.npy"),
        &[exp.shots.len(), exp.steps, n_receivers],
        &traces,
    )?;
    if let (Some(every), Some(background_frames)) = (cli.every, &frames) {
        let shape = [background_frames.len() / n_cells, exp.nz, exp.nx];
        npy::write_f32(&cli.out.join("wavefield.npy"), &shape, background_frames)?;

        // Echo: the same shot in background plus perturbation, minus the
        // background recording.
        let mut perturbed_traces = vec![0.0; exp.steps * n_receivers];
        let perturbed_frames = solver.run(
            exp.shots[0],
            &Model::new(&exp, true),
            Some(every),
            &mut perturbed_traces,
        );
        let echo: Vec<f32> = perturbed_frames
            .iter()
            .zip(background_frames)
            .map(|(&p, &b)| p - b)
            .collect();
        npy::write_f32(&cli.out.join("echo.npy"), &shape, &echo)?;
    }

    let experiment_file = cli.experiment.to_string_lossy();
    let run_json = output::run_json(&exp, &experiment_file, cli.every);
    std::fs::write(
        cli.out.join("run.json"),
        serde_json::to_string_pretty(&run_json).unwrap(),
    )
    .map_err(|e| format!("cannot write run.json: {e}"))?;
    let result_json = output::result_json(&exp, mode_name(cli.mode));
    std::fs::write(
        cli.out.join("result.json"),
        serde_json::to_string_pretty(&result_json).unwrap(),
    )
    .map_err(|e| format!("cannot write result.json: {e}"))?;

    println!("shot\tmode\tl2");
    for s in 0..exp.shots.len() {
        let start = s * exp.steps * n_receivers;
        let l2 = traces[start..start + exp.steps * n_receivers]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        println!("{s}\tforward\t{l2:.12e}");
    }
    Ok(())
}
