use crate::{lennard_jones_pair_energy, lennard_jones_pair_force, Integrator};
use plotters::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct MdError(pub String);

impl fmt::Display for MdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for MdError {}

impl From<std::io::Error> for MdError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for MdError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeriodicBox {
    pub lengths: [f64; 2],
}

impl PeriodicBox {
    pub fn new(lengths: [f64; 2]) -> Result<Self, MdError> {
        if lengths
            .iter()
            .all(|length| length.is_finite() && *length > 0.0)
        {
            Ok(Self { lengths })
        } else {
            Err(MdError("box lengths must be positive and finite".into()))
        }
    }

    pub fn area(&self) -> f64 {
        self.lengths[0] * self.lengths[1]
    }

    pub fn wrap(&self, position: [f64; 2]) -> [f64; 2] {
        [
            position[0].rem_euclid(self.lengths[0]),
            position[1].rem_euclid(self.lengths[1]),
        ]
    }

    pub fn contains(&self, position: [f64; 2]) -> bool {
        position[0] >= 0.0
            && position[0] < self.lengths[0]
            && position[1] >= 0.0
            && position[1] < self.lengths[1]
    }

    pub fn minimum_image(&self, from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
        let mut displacement = [to[0] - from[0], to[1] - from[1]];
        for (component, length) in displacement.iter_mut().zip(self.lengths) {
            *component -= length * (*component / length).round();
        }
        displacement
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairInteraction {
    pub energy: f64,
    pub radial_force: f64,
}

impl PairInteraction {
    pub const ZERO: Self = Self {
        energy: 0.0,
        radial_force: 0.0,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct ShiftedLennardJones {
    pub cutoff: f64,
    cutoff_energy: f64,
}

impl ShiftedLennardJones {
    pub fn new(cutoff: f64) -> Result<Self, MdError> {
        if !cutoff.is_finite() || cutoff <= 0.0 {
            return Err(MdError("cutoff must be positive and finite".into()));
        }
        Ok(Self {
            cutoff,
            cutoff_energy: lennard_jones_pair_energy(cutoff),
        })
    }

    pub fn pair(&self, distance: f64) -> Result<PairInteraction, MdError> {
        if !distance.is_finite() || distance <= 0.0 {
            return Err(MdError("pair distance must be positive and finite".into()));
        }
        if distance >= self.cutoff {
            return Ok(PairInteraction::ZERO);
        }
        Ok(PairInteraction {
            energy: lennard_jones_pair_energy(distance) - self.cutoff_energy,
            radial_force: lennard_jones_pair_force(distance),
        })
    }
}

#[derive(Clone, Debug)]
pub struct FluidState {
    pub positions: Vec<[f64; 2]>,
    pub velocities: Vec<[f64; 2]>,
    pub forces: Vec<[f64; 2]>,
    pub cell: PeriodicBox,
}

impl FluidState {
    pub fn new(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        cell: PeriodicBox,
    ) -> Result<Self, MdError> {
        if positions.is_empty() || positions.len() != velocities.len() {
            return Err(MdError(
                "positions and velocities must have the same nonzero length".into(),
            ));
        }
        if !positions.iter().all(|position| cell.contains(*position)) {
            return Err(MdError(
                "all positions must be inside the periodic box".into(),
            ));
        }
        let forces = vec![[0.0; 2]; positions.len()];
        Ok(Self {
            positions,
            velocities,
            forces,
            cell,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ForceEvaluation {
    pub forces: Vec<[f64; 2]>,
    pub potential_energy: f64,
    pub minimum_distance: f64,
}

pub fn evaluate(
    state: &FluidState,
    potential: ShiftedLennardJones,
) -> Result<ForceEvaluation, MdError> {
    let mut forces = vec![[0.0; 2]; state.positions.len()];
    let mut potential_energy = 0.0;
    let mut minimum_distance = f64::INFINITY;
    for first in 0..state.positions.len() - 1 {
        for second in first + 1..state.positions.len() {
            let displacement = state
                .cell
                .minimum_image(state.positions[first], state.positions[second]);
            let distance = displacement[0].hypot(displacement[1]);
            minimum_distance = minimum_distance.min(distance);
            let pair = potential.pair(distance)?;
            potential_energy += pair.energy;
            if pair.radial_force != 0.0 {
                for component in 0..2 {
                    let force = pair.radial_force * displacement[component] / distance;
                    forces[first][component] -= force;
                    forces[second][component] += force;
                }
            }
        }
    }
    Ok(ForceEvaluation {
        forces,
        potential_energy,
        minimum_distance,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FluidConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub cutoff: f64,
    pub rescale_every: usize,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2_000,
            steps: 10_000,
            sample_every: 50,
            seed: 2_026,
            cutoff: 2.5,
            rescale_every: 50,
        }
    }
}

pub fn triangular_lattice(n: usize, rho: f64) -> Result<(PeriodicBox, Vec<[f64; 2]>), MdError> {
    if n == 0 || !rho.is_finite() || rho <= 0.0 {
        return Err(MdError("n and rho must be positive".into()));
    }
    let mut factors = (0..=n)
        .filter(|rows| *rows > 0 && rows.is_multiple_of(2) && n.is_multiple_of(*rows))
        .map(|rows| (n / rows, rows))
        .collect::<Vec<_>>();
    factors.sort_by_key(|(columns, rows)| columns.abs_diff(*rows));
    let (columns, rows) = factors.first().copied().ok_or_else(|| {
        MdError("n must permit a rectangular triangular lattice with an even row count".into())
    })?;
    let spacing = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let cell = PeriodicBox::new([
        columns as f64 * spacing,
        rows as f64 * 3.0_f64.sqrt() * spacing / 2.0,
    ])?;
    let mut positions = Vec::with_capacity(n);
    for row in 0..rows {
        for column in 0..columns {
            positions.push([
                (column as f64 + 0.5 * (row % 2) as f64) * spacing,
                row as f64 * 3.0_f64.sqrt() * spacing / 2.0,
            ]);
        }
    }
    Ok((cell, positions))
}

pub fn kinetic_energy(state: &FluidState) -> f64 {
    0.5 * state
        .velocities
        .iter()
        .map(|velocity| velocity[0].powi(2) + velocity[1].powi(2))
        .sum::<f64>()
}

pub fn temperature(state: &FluidState) -> f64 {
    kinetic_energy(state) / state.positions.len() as f64
}

pub fn total_momentum(state: &FluidState) -> [f64; 2] {
    state.velocities.iter().fold([0.0; 2], |sum, velocity| {
        [sum[0] + velocity[0], sum[1] + velocity[1]]
    })
}

pub fn rescale_temperature(state: &mut FluidState, target: f64) -> Result<(), MdError> {
    let current = temperature(state);
    if !target.is_finite() || target <= 0.0 || current <= 0.0 {
        return Err(MdError("temperature must be positive and finite".into()));
    }
    let scale = (target / current).sqrt();
    for velocity in &mut state.velocities {
        velocity[0] *= scale;
        velocity[1] *= scale;
    }
    Ok(())
}

pub fn initialize_fluid(config: &FluidConfig) -> Result<FluidState, MdError> {
    validate_config(config)?;
    let (cell, positions) = triangular_lattice(config.n, config.rho)?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut velocities = (0..config.n)
        .map(|_| {
            [
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            ]
        })
        .collect::<Vec<_>>();
    let mean = velocities.iter().fold([0.0; 2], |sum, velocity| {
        [sum[0] + velocity[0], sum[1] + velocity[1]]
    });
    for velocity in &mut velocities {
        velocity[0] -= mean[0] / config.n as f64;
        velocity[1] -= mean[1] / config.n as f64;
    }
    let mut state = FluidState::new(positions, velocities, cell)?;
    rescale_temperature(&mut state, config.temperature)?;
    state.forces = evaluate(&state, ShiftedLennardJones::new(config.cutoff)?)?.forces;
    Ok(state)
}

fn validate_config(config: &FluidConfig) -> Result<(), MdError> {
    if config.n < 2
        || config.dt <= 0.0
        || config.steps == 0
        || config.sample_every == 0
        || config.rescale_every == 0
        || !config.steps.is_multiple_of(config.sample_every)
    {
        return Err(MdError("invalid run parameters or sampling cadence".into()));
    }
    let (cell, _) = triangular_lattice(config.n, config.rho)?;
    if config.cutoff > 0.5 * cell.lengths[0].min(cell.lengths[1]) {
        return Err(MdError(
            "cutoff exceeds half the shortest box length".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct FluidVelocityVerlet {
    potential: ShiftedLennardJones,
}

impl FluidVelocityVerlet {
    pub fn new(cutoff: f64) -> Result<Self, MdError> {
        Ok(Self {
            potential: ShiftedLennardJones::new(cutoff)?,
        })
    }
}

impl Integrator<FluidState> for FluidVelocityVerlet {
    fn step(&self, state: &mut FluidState, dt: f64) {
        for (velocity, force) in state.velocities.iter_mut().zip(&state.forces) {
            velocity[0] += 0.5 * dt * force[0];
            velocity[1] += 0.5 * dt * force[1];
        }
        for (position, velocity) in state.positions.iter_mut().zip(&state.velocities) {
            *position = state.cell.wrap([
                position[0] + dt * velocity[0],
                position[1] + dt * velocity[1],
            ]);
        }
        let new_forces = evaluate(state, self.potential)
            .expect("valid state during integration")
            .forces;
        for (velocity, force) in state.velocities.iter_mut().zip(&new_forces) {
            velocity[0] += 0.5 * dt * force[0];
            velocity[1] += 0.5 * dt * force[1];
        }
        state.forces = new_forces;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub format: String,
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    #[serde(rename = "box")]
    pub box_lengths: [f64; 2],
    pub integrator: String,
    pub potential: String,
    pub cutoff: f64,
    pub equilibration_rescale_every: usize,
    pub production_reference_energy: f64,
    pub saved_frames: usize,
}

impl RunMetadata {
    pub fn for_test(n: usize, box_lengths: [f64; 2], rho: f64) -> Self {
        Self {
            format: "amat5315-md-v1".into(),
            n,
            rho,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 0,
            steps: 1,
            sample_every: 1,
            seed: 2026,
            box_lengths,
            integrator: "velocity-verlet".into(),
            potential: "lennard-jones-potential-shifted".into(),
            cutoff: 2.5,
            equilibration_rescale_every: 50,
            production_reference_energy: 0.0,
            saved_frames: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryFrame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

#[derive(Clone, Debug)]
pub struct SimulationOutput {
    pub metadata: RunMetadata,
    pub frames: Vec<TrajectoryFrame>,
}

pub fn run_fluid(config: &FluidConfig) -> Result<SimulationOutput, MdError> {
    let mut state = initialize_fluid(config)?;
    let integrator = FluidVelocityVerlet::new(config.cutoff)?;
    for step in 1..=config.eq_steps {
        integrator.step(&mut state, config.dt);
        if step % config.rescale_every == 0 {
            rescale_temperature(&mut state, config.temperature)?;
        }
    }
    let potential = ShiftedLennardJones::new(config.cutoff)?;
    let initial_potential = evaluate(&state, potential)?.potential_energy;
    let reference_energy = kinetic_energy(&state) + initial_potential;
    let mut frames = Vec::with_capacity(config.steps / config.sample_every);
    for step in 1..=config.steps {
        integrator.step(&mut state, config.dt);
        if step % config.sample_every == 0 {
            let e_pot = evaluate(&state, potential)?.potential_energy;
            frames.push(TrajectoryFrame {
                step,
                t: step as f64 * config.dt,
                pos: state.positions.clone(),
                vel: state.velocities.clone(),
                e_pot,
                e_kin: kinetic_energy(&state),
            });
        }
    }
    let metadata = RunMetadata {
        format: "amat5315-md-v1".into(),
        n: config.n,
        rho: config.rho,
        temperature: config.temperature,
        dt: config.dt,
        eq_steps: config.eq_steps,
        steps: config.steps,
        sample_every: config.sample_every,
        seed: config.seed,
        box_lengths: state.cell.lengths,
        integrator: "velocity-verlet".into(),
        potential: "lennard-jones-potential-shifted".into(),
        cutoff: config.cutoff,
        equilibration_rescale_every: config.rescale_every,
        production_reference_energy: reference_energy,
        saved_frames: frames.len(),
    };
    Ok(SimulationOutput { metadata, frames })
}

#[derive(Clone, Debug)]
pub struct RadialDistribution {
    pub radii: Vec<f64>,
    pub values: Vec<f64>,
}

pub fn radial_distribution(
    frames: &[TrajectoryFrame],
    metadata: &RunMetadata,
    bins: usize,
) -> Result<RadialDistribution, MdError> {
    if frames.is_empty() || bins == 0 {
        return Err(MdError("RDF requires frames and bins".into()));
    }
    let cell = PeriodicBox::new(metadata.box_lengths)?;
    let rmax = 0.5 * cell.lengths[0].min(cell.lengths[1]);
    let dr = rmax / bins as f64;
    let mut counts = vec![0.0; bins];
    for frame in frames {
        for first in 0..frame.pos.len() - 1 {
            for second in first + 1..frame.pos.len() {
                let d = cell.minimum_image(frame.pos[first], frame.pos[second]);
                let r = d[0].hypot(d[1]);
                if r < rmax {
                    counts[(r / dr) as usize] += 1.0;
                }
            }
        }
    }
    let mut radii = Vec::with_capacity(bins);
    let mut values = Vec::with_capacity(bins);
    for (bin, count) in counts.into_iter().enumerate() {
        let inner = bin as f64 * dr;
        let outer = inner + dr;
        radii.push(0.5 * (inner + outer));
        values.push(
            count
                / (frames.len() as f64
                    * metadata.n as f64
                    * metadata.rho
                    * PI
                    * (outer * outer - inner * inner)
                    / 2.0),
        );
    }
    Ok(RadialDistribution { radii, values })
}

pub fn write_artifacts(output: &Path, simulation: &SimulationOutput) -> Result<(), MdError> {
    fs::create_dir_all(output)?;
    let run_path = output.join("run.json");
    let trajectory_path = output.join("traj.jsonl");
    if run_path.exists() || trajectory_path.exists() {
        return Err(MdError(format!(
            "refusing to overwrite artifacts in {}",
            output.display()
        )));
    }
    let run_tmp = output.join("run.json.tmp");
    let trajectory_tmp = output.join("traj.jsonl.tmp");
    let result = (|| {
        serde_json::to_writer_pretty(
            BufWriter::new(File::create(&run_tmp)?),
            &simulation.metadata,
        )?;
        let mut writer = BufWriter::new(File::create(&trajectory_tmp)?);
        for frame in &simulation.frames {
            serde_json::to_writer(&mut writer, frame)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        fs::rename(&run_tmp, &run_path)?;
        fs::rename(&trajectory_tmp, &trajectory_path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(run_tmp);
        let _ = fs::remove_file(trajectory_tmp);
    }
    result
}

pub fn load_artifacts(input: &Path) -> Result<(RunMetadata, Vec<TrajectoryFrame>), MdError> {
    let metadata = serde_json::from_reader(BufReader::new(File::open(input.join("run.json"))?))?;
    let reader = BufReader::new(File::open(input.join("traj.jsonl"))?);
    let mut frames = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        frames.push(
            serde_json::from_str(&line)
                .map_err(|error| MdError(format!("traj.jsonl line {}: {error}", index + 1)))?,
        );
    }
    Ok((metadata, frames))
}

#[derive(Clone, Debug)]
pub struct CheckReport {
    pub energy_integrity: bool,
    pub momentum_conservation: bool,
    pub energy_drift: bool,
    pub maximum_energy_mismatch: f64,
    pub maximum_momentum_per_particle: f64,
    pub rolling_energy_drift: f64,
    pub maximum_relative_energy_error: f64,
    pub mean_temperature: f64,
    pub minimum_pair_distance: f64,
}

impl CheckReport {
    pub fn passed(&self) -> bool {
        self.energy_integrity && self.momentum_conservation && self.energy_drift
    }
}

pub fn check_artifacts(input: &Path) -> Result<CheckReport, MdError> {
    let (metadata, frames) = load_artifacts(input)?;
    if metadata.format != "amat5315-md-v1"
        || frames.len() != metadata.saved_frames
        || frames.is_empty()
    {
        return Err(MdError(
            "metadata format or saved frame count is invalid".into(),
        ));
    }
    let cell = PeriodicBox::new(metadata.box_lengths)?;
    let potential = ShiftedLennardJones::new(metadata.cutoff)?;
    let mut maximum_energy_mismatch: f64 = 0.0;
    let mut maximum_momentum_per_particle: f64 = 0.0;
    let mut maximum_relative_energy_error: f64 = 0.0;
    let mut minimum_pair_distance = f64::INFINITY;
    let mut temperatures = Vec::with_capacity(frames.len());
    let mut energies = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let expected_step = (index + 1) * metadata.sample_every;
        if frame.step != expected_step
            || (frame.t - frame.step as f64 * metadata.dt).abs() > 1.0e-10
        {
            return Err(MdError(format!("invalid cadence at frame {}", index + 1)));
        }
        if frame.pos.len() != metadata.n
            || frame.vel.len() != metadata.n
            || !frame.pos.iter().all(|position| cell.contains(*position))
        {
            return Err(MdError(format!(
                "invalid particle data at frame {}",
                index + 1
            )));
        }
        let state = FluidState::new(frame.pos.clone(), frame.vel.clone(), cell)?;
        let evaluated = evaluate(&state, potential)?;
        let e_kin = kinetic_energy(&state);
        let tolerance = 1.0e-9
            * 1.0_f64
                .max(evaluated.potential_energy.abs())
                .max(e_kin.abs());
        let mismatch = (frame.e_pot - evaluated.potential_energy)
            .abs()
            .max((frame.e_kin - e_kin).abs());
        maximum_energy_mismatch = maximum_energy_mismatch.max(mismatch / tolerance);
        let momentum = total_momentum(&state);
        maximum_momentum_per_particle =
            maximum_momentum_per_particle.max(momentum[0].hypot(momentum[1]) / metadata.n as f64);
        let total = evaluated.potential_energy + e_kin;
        energies.push(total);
        temperatures.push(e_kin / metadata.n as f64);
        maximum_relative_energy_error = maximum_relative_energy_error.max(
            (total - metadata.production_reference_energy).abs()
                / metadata.production_reference_energy.abs(),
        );
        minimum_pair_distance = minimum_pair_distance.min(evaluated.minimum_distance);
    }
    let window = 20.min(energies.len());
    let means = energies
        .windows(window)
        .map(|values| values.iter().sum::<f64>() / window as f64)
        .collect::<Vec<_>>();
    let min_mean = means.iter().copied().fold(f64::INFINITY, f64::min);
    let max_mean = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rolling_energy_drift = (max_mean - min_mean) / metadata.production_reference_energy.abs();
    Ok(CheckReport {
        energy_integrity: maximum_energy_mismatch <= 1.0,
        momentum_conservation: maximum_momentum_per_particle < 1.0e-10,
        energy_drift: rolling_energy_drift < 1.0e-3,
        maximum_energy_mismatch,
        maximum_momentum_per_particle,
        rolling_energy_drift,
        maximum_relative_energy_error,
        mean_temperature: temperatures.iter().sum::<f64>() / temperatures.len() as f64,
        minimum_pair_distance,
    })
}

pub fn render_video_frame(
    frame: &TrajectoryFrame,
    rdf: &RadialDistribution,
    metadata: &RunMetadata,
    dimensions: [u32; 2],
) -> Result<Vec<u8>, MdError> {
    let mut buffer = vec![255; dimensions[0] as usize * dimensions[1] as usize * 3];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (dimensions[0], dimensions[1]))
            .into_drawing_area();
        root.fill(&WHITE)
            .map_err(|error| MdError(error.to_string()))?;
        let (particles, structure) = root.split_horizontally(dimensions[0] / 2);
        let mut particle_chart = ChartBuilder::on(&particles)
            .caption(
                format!("Lennard-Jones fluid: step {}, t={:.2}", frame.step, frame.t),
                ("sans-serif", 24),
            )
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..metadata.box_lengths[0], 0.0..metadata.box_lengths[1])
            .map_err(|error| MdError(error.to_string()))?;
        particle_chart
            .configure_mesh()
            .x_desc("x / sigma")
            .y_desc("y / sigma")
            .draw()
            .map_err(|error| MdError(error.to_string()))?;
        let maximum_speed = frame
            .vel
            .iter()
            .map(|velocity| velocity[0].hypot(velocity[1]))
            .fold(0.0, f64::max)
            .max(1.0e-12);
        particle_chart
            .draw_series(
                frame
                    .pos
                    .iter()
                    .zip(&frame.vel)
                    .map(|(position, velocity)| {
                        let fraction = velocity[0].hypot(velocity[1]) / maximum_speed;
                        Circle::new(
                            (position[0], position[1]),
                            5,
                            HSLColor(0.65 * (1.0 - fraction), 0.8, 0.5).filled(),
                        )
                    }),
            )
            .map_err(|error| MdError(error.to_string()))?;

        let maximum_g = rdf.values.iter().copied().fold(1.0, f64::max) * 1.1;
        let rmax = rdf.radii.last().copied().unwrap_or(1.0);
        let mut rdf_chart = ChartBuilder::on(&structure)
            .caption("Radial distribution function", ("sans-serif", 24))
            .margin(15)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(0.0..rmax, 0.0..maximum_g)
            .map_err(|error| MdError(error.to_string()))?;
        rdf_chart
            .configure_mesh()
            .x_desc("r / sigma")
            .y_desc("g(r)")
            .draw()
            .map_err(|error| MdError(error.to_string()))?;
        rdf_chart
            .draw_series(LineSeries::new(
                rdf.radii.iter().copied().zip(rdf.values.iter().copied()),
                &BLUE,
            ))
            .map_err(|error| MdError(error.to_string()))?;
        root.present().map_err(|error| MdError(error.to_string()))?;
    }
    Ok(buffer)
}

pub fn encode_video(
    input: &Path,
    output: &Path,
    fps: u32,
    rdf_bins: usize,
    rdf_window: usize,
) -> Result<(), MdError> {
    if fps == 0 || rdf_bins == 0 || rdf_window == 0 {
        return Err(MdError(
            "fps, RDF bins, and RDF window must be positive".into(),
        ));
    }
    if output.exists() {
        return Err(MdError(format!(
            "refusing to overwrite {}",
            output.display()
        )));
    }
    let (metadata, frames) = load_artifacts(input)?;
    let dimensions = [1280_u32, 720_u32];
    let size = format!("{}x{}", dimensions[0], dimensions[1]);
    let rate = fps.to_string();
    let mut child = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &size,
            "-framerate",
            &rate,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(output)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|error| MdError(format!("could not start ffmpeg: {error}")))?;
    let result = (|| {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| MdError("ffmpeg stdin unavailable".into()))?;
        for (index, frame) in frames.iter().enumerate() {
            let start = (index + 1).saturating_sub(rdf_window);
            let rdf = radial_distribution(&frames[start..=index], &metadata, rdf_bins)?;
            stdin.write_all(&render_video_frame(frame, &rdf, &metadata, dimensions)?)?;
        }
        Ok::<(), MdError>(())
    })();
    drop(child.stdin.take());
    let status = child.wait()?;
    if result.is_err() || !status.success() {
        let _ = fs::remove_file(output);
        return result.and_then(|_| Err(MdError(format!("ffmpeg exited with {status}"))));
    }
    Ok(())
}
