use md::fluid::{initialize_fluid, run_fluid, temperature, total_momentum, FluidConfig};

#[test]
fn default_initialization_is_reproducible_and_prepared() {
    let first = initialize_fluid(&FluidConfig::default()).unwrap();
    let second = initialize_fluid(&FluidConfig::default()).unwrap();
    assert_eq!(first.positions, second.positions);
    assert_eq!(first.velocities, second.velocities);
    assert!((first.positions.len() as f64 / first.cell.area() - 0.8).abs() < 1.0e-12);
    let momentum = total_momentum(&first);
    assert!(momentum[0].hypot(momentum[1]) < 1.0e-12);
    assert!((temperature(&first) - 0.5).abs() < 1.0e-12);
}

#[test]
fn sampling_excludes_step_zero_and_uses_production_steps() {
    let config = FluidConfig {
        n: 36,
        eq_steps: 10,
        steps: 20,
        sample_every: 5,
        rescale_every: 5,
        ..FluidConfig::default()
    };
    let output = run_fluid(&config).unwrap();
    assert_eq!(
        output
            .frames
            .iter()
            .map(|frame| frame.step)
            .collect::<Vec<_>>(),
        [5, 10, 15, 20]
    );
    assert_eq!(output.frames[0].t, 0.05);
    assert_eq!(output.metadata.saved_frames, 4);
}

#[test]
fn production_temperature_can_ramp_linearly_to_the_final_target() {
    let config = FluidConfig {
        n: 36,
        eq_steps: 0,
        steps: 10,
        sample_every: 1,
        temperature: 0.5,
        ramp_to: Some(1.0),
        ..FluidConfig::default()
    };
    let output = run_fluid(&config).unwrap();
    assert_eq!(output.metadata.ramp_to, Some(1.0));
    assert!((output.frames.first().unwrap().e_kin / 36.0 - 0.55).abs() < 1.0e-10);
    assert!((output.frames.last().unwrap().e_kin / 36.0 - 1.0).abs() < 1.0e-10);
}
