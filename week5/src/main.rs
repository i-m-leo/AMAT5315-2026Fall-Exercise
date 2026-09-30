//! `seismic` simulates acoustic waves and their derivatives.
//!
//! ```text
//! seismic --experiment inputs/reflector.json --mode forward --every 3 \
//!         --out artifacts/forward
//! seismic --experiment inputs/reflector.json --mode born --out artifacts/born
//! seismic --experiment inputs/reflector.json --mode adjoint \
//!         --data artifacts/born/born_data.npy --out artifacts/adjoint
//! ```

use clap::{Parser, ValueEnum};
use seismic::npy;
use seismic::output;
use seismic::{modes, Experiment, Model, Solver};
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
    /// adjoint only: born_data.npy of a born run; the receiver weights
    #[arg(long)]
    data: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
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
    std::fs::create_dir_all(&cli.out)
        .map_err(|e| format!("cannot create {}: {e}", cli.out.display()))?;
    let experiment_file = cli.experiment.to_string_lossy();

    match cli.mode {
        Mode::Forward => forward(cli, &exp, &experiment_file),
        Mode::Born => born(cli, &exp, &experiment_file),
        Mode::Adjoint => adjoint(cli, &exp, &experiment_file),
    }
}

/// Write `run.json` and `result.json`, and print the per-shot L2 table.
fn finish(
    cli: &Cli,
    exp: &Experiment,
    experiment_file: &str,
    values: &[f64],
    statistics: Option<modes::AdjointStats>,
) -> Result<(), String> {
    let adjoint = matches!(cli.mode, Mode::Adjoint);
    let run_json = output::run_json(exp, experiment_file, cli.every, adjoint);
    std::fs::write(
        cli.out.join("run.json"),
        serde_json::to_string_pretty(&run_json).unwrap(),
    )
    .map_err(|e| format!("cannot write run.json: {e}"))?;
    let result_json = output::result_json_with(exp, mode_name(cli.mode), statistics);
    std::fs::write(
        cli.out.join("result.json"),
        serde_json::to_string_pretty(&result_json).unwrap(),
    )
    .map_err(|e| format!("cannot write result.json: {e}"))?;

    let mode = mode_name(cli.mode);
    println!("shot\tmode\tl2");
    for (s, l2) in values.iter().enumerate() {
        println!("{s}\t{mode}\t{l2:.12e}");
    }
    Ok(())
}

/// Per-shot L2 norms of a `[shot][step][receiver]` array.
fn per_shot_l2(exp: &Experiment, data: &[f64]) -> Vec<f64> {
    let width = exp.steps * exp.receivers.len();
    data.chunks(width)
        .map(|shot| shot.iter().map(|v| v * v).sum::<f64>().sqrt())
        .collect()
}

fn forward(cli: &Cli, exp: &Experiment, experiment_file: &str) -> Result<(), String> {
    let solver = Solver::new(exp);
    let n_receivers = exp.receivers.len();
    let n_cells = exp.nx * exp.nz;
    let background = Model::new(exp, false);

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
            &Model::new(exp, true),
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

    let values = per_shot_l2(exp, &traces);
    finish(cli, exp, experiment_file, &values, None)
}

fn born(cli: &Cli, exp: &Experiment, experiment_file: &str) -> Result<(), String> {
    if cli.every.is_some() {
        return Err("--every is not used by born mode".into());
    }
    let data = modes::born(exp)?;
    npy::write_f64(
        &cli.out.join("born_data.npy"),
        &[exp.shots.len(), exp.steps, exp.receivers.len()],
        &data,
    )?;
    let values = per_shot_l2(exp, &data);
    finish(cli, exp, experiment_file, &values, None)
}

fn adjoint(cli: &Cli, exp: &Experiment, experiment_file: &str) -> Result<(), String> {
    let data_path = cli
        .data
        .as_ref()
        .ok_or("adjoint mode requires --data <born_data.npy>")?;
    let expected = [exp.shots.len(), exp.steps, exp.receivers.len()];
    let data = npy::read_f64(data_path, &expected)?;

    let (image, statistics, frames) = modes::adjoint(exp, &data, modes::Storage::Full, cli.every)?;
    npy::write_f64(&cli.out.join("image.npy"), &[exp.nz, exp.nx], &image)?;
    if cli.every.is_some() {
        let n_cells = exp.nx * exp.nz;
        let shape = [frames.len() / n_cells, exp.nz, exp.nx];
        npy::write_f32(&cli.out.join("wavefield.npy"), &shape, &frames)?;
    }

    let values = per_shot_l2(exp, &data);
    finish(cli, exp, experiment_file, &values, Some(statistics))
}
