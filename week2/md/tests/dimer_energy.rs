use md::{
    lennard_jones_pair_energy, run_dimer_experiment, DimerState, EnergySample, ForwardEuler,
    TimeIntegrator, VelocityVerlet,
};

const TIME_STEP: f64 = 0.01;
const SHORT_STEPS: usize = 500;
const LONG_STEPS: usize = 5_000;

fn accepts_integrator<I: md::Integrator<DimerState>>(_: &I) {}

fn run_with<I: TimeIntegrator>(integrator: &I, steps: usize) -> Vec<EnergySample> {
    run_dimer_experiment(integrator, DimerState::stationary(1.2), TIME_STEP, steps)
}

fn maximum_error(samples: &[EnergySample]) -> f64 {
    samples
        .iter()
        .map(|sample| sample.relative_energy_error.abs())
        .fold(0.0, f64::max)
}

#[test]
fn dimer_energy_error_distinguishes_euler_from_verlet() {
    let euler = run_with(&ForwardEuler, SHORT_STEPS);
    let verlet = run_with(&VelocityVerlet, SHORT_STEPS);
    let long_verlet = run_with(&VelocityVerlet, LONG_STEPS);
    let initial_energy = lennard_jones_pair_energy(1.2);

    assert_eq!(euler.len(), SHORT_STEPS + 1);
    assert_eq!(verlet.len(), SHORT_STEPS + 1);
    assert_eq!(long_verlet.len(), LONG_STEPS + 1);
    assert_eq!(euler[0].total_energy, initial_energy);
    assert_eq!(verlet[0].total_energy, initial_energy);
    assert_eq!(euler[0].relative_energy_error, 0.0);
    assert_eq!(verlet[0].relative_energy_error, 0.0);

    assert!(verlet
        .iter()
        .all(|sample| sample.relative_energy_error.is_finite()));
    assert!(maximum_error(&verlet) < 1.0e-3);
    assert!(euler.last().unwrap().relative_energy_error > 0.5);
    assert!(maximum_error(&long_verlet) < 1.0e-3);
}

#[test]
fn both_dimer_methods_share_the_generic_trait() {
    accepts_integrator(&ForwardEuler);
    accepts_integrator(&VelocityVerlet);
}
