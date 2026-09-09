use crate::{lennard_jones_pair_energy, lennard_jones_pair_force};

/// Positions, velocities, and masses for two atoms moving in two dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct DimerState {
    pub positions: [[f64; 2]; 2],
    pub velocities: [[f64; 2]; 2],
    pub masses: [f64; 2],
}

impl DimerState {
    /// Creates two stationary unit-mass atoms centered on the x axis.
    pub fn stationary(separation: f64) -> Self {
        Self {
            positions: [[-0.5 * separation, 0.0], [0.5 * separation, 0.0]],
            velocities: [[0.0; 2]; 2],
            masses: [1.0; 2],
        }
    }
}

/// Advances a state by one timestep.
pub trait Integrator<S> {
    fn step(&self, state: &mut S, time_step: f64);
}

/// Compatibility marker for integrators that advance a dimer.
pub trait TimeIntegrator: Integrator<DimerState> {}

impl<T: Integrator<DimerState>> TimeIntegrator for T {}

#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;

impl Integrator<DimerState> for ForwardEuler {
    fn step(&self, state: &mut DimerState, time_step: f64) {
        let acceleration = accelerations(state);
        for ((position, velocity), atom_acceleration) in state
            .positions
            .iter_mut()
            .zip(&mut state.velocities)
            .zip(acceleration)
        {
            for ((coordinate, speed), component_acceleration) in
                position.iter_mut().zip(velocity).zip(atom_acceleration)
            {
                *coordinate += *speed * time_step;
                *speed += component_acceleration * time_step;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VelocityVerlet;

impl Integrator<DimerState> for VelocityVerlet {
    fn step(&self, state: &mut DimerState, time_step: f64) {
        let old_acceleration = accelerations(state);
        for ((position, velocity), atom_acceleration) in state
            .positions
            .iter_mut()
            .zip(&state.velocities)
            .zip(old_acceleration)
        {
            for ((coordinate, speed), component_acceleration) in
                position.iter_mut().zip(velocity).zip(atom_acceleration)
            {
                *coordinate +=
                    *speed * time_step + 0.5 * component_acceleration * time_step.powi(2);
            }
        }

        let new_acceleration = accelerations(state);
        for ((velocity, old_atom_acceleration), new_atom_acceleration) in state
            .velocities
            .iter_mut()
            .zip(old_acceleration)
            .zip(new_acceleration)
        {
            for ((speed, old_component_acceleration), new_component_acceleration) in velocity
                .iter_mut()
                .zip(old_atom_acceleration)
                .zip(new_atom_acceleration)
            {
                *speed +=
                    0.5 * (old_component_acceleration + new_component_acceleration) * time_step;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnergySample {
    pub step: usize,
    pub time: f64,
    pub total_energy: f64,
    pub relative_energy_error: f64,
}

/// Runs identical initial conditions with any integrator and records energy drift.
pub fn run_dimer_experiment<I: TimeIntegrator>(
    integrator: &I,
    mut state: DimerState,
    time_step: f64,
    steps: usize,
) -> Vec<EnergySample> {
    let initial_energy = total_energy(&state);
    let mut samples = Vec::with_capacity(steps + 1);
    samples.push(energy_sample(0, time_step, &state, initial_energy));

    for step in 1..=steps {
        integrator.step(&mut state, time_step);
        samples.push(energy_sample(step, time_step, &state, initial_energy));
    }
    samples
}

fn accelerations(state: &DimerState) -> [[f64; 2]; 2] {
    let displacement = [
        state.positions[1][0] - state.positions[0][0],
        state.positions[1][1] - state.positions[0][1],
    ];
    let distance = displacement[0].hypot(displacement[1]);
    assert!(distance > 0.0, "atoms cannot occupy the same position");
    let radial_force = lennard_jones_pair_force(distance);
    let force_on_second = [
        radial_force * displacement[0] / distance,
        radial_force * displacement[1] / distance,
    ];

    [
        [
            -force_on_second[0] / state.masses[0],
            -force_on_second[1] / state.masses[0],
        ],
        [
            force_on_second[0] / state.masses[1],
            force_on_second[1] / state.masses[1],
        ],
    ]
}

fn total_energy(state: &DimerState) -> f64 {
    let kinetic_energy: f64 = state
        .velocities
        .iter()
        .zip(state.masses)
        .map(|(velocity, mass)| 0.5 * mass * (velocity[0].powi(2) + velocity[1].powi(2)))
        .sum();
    let separation = [
        state.positions[1][0] - state.positions[0][0],
        state.positions[1][1] - state.positions[0][1],
    ];
    kinetic_energy + lennard_jones_pair_energy(separation[0].hypot(separation[1]))
}

fn energy_sample(
    step: usize,
    time_step: f64,
    state: &DimerState,
    initial_energy: f64,
) -> EnergySample {
    let energy = total_energy(state);
    EnergySample {
        step,
        time: step as f64 * time_step,
        total_energy: energy,
        relative_energy_error: (energy - initial_energy) / initial_energy.abs(),
    }
}
