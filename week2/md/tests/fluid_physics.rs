use md::fluid::{evaluate, FluidState, PeriodicBox, ShiftedLennardJones};

fn close(left: f64, right: f64) {
    assert!((left - right).abs() < 1.0e-10, "{left} != {right}");
}

#[test]
fn periodic_box_wraps_and_uses_the_minimum_image() {
    let cell = PeriodicBox::new([10.0, 8.0]).unwrap();
    let wrapped = cell.wrap([-0.2, 8.1]);
    close(wrapped[0], 9.8);
    close(wrapped[1], 0.1);
    let displacement = cell.minimum_image([9.8, 0.2], [0.3, 7.9]);
    close(displacement[0], 0.5);
    close(displacement[1], -0.3);
}

#[test]
fn potential_is_shifted_to_zero_at_the_cutoff() {
    let potential = ShiftedLennardJones::new(2.5).unwrap();
    assert_eq!(potential.pair(2.5).unwrap().energy, 0.0);
    assert_eq!(potential.pair(3.0).unwrap().radial_force, 0.0);
    assert!(potential.pair(2.5 - 1.0e-9).unwrap().energy.abs() < 1.0e-8);
    close(
        potential.pair(1.2).unwrap().radial_force,
        md::lennard_jones_pair_force(1.2),
    );
}

#[test]
fn forces_on_all_atoms_sum_to_zero() {
    let cell = PeriodicBox::new([8.0, 8.0]).unwrap();
    let state = FluidState::new(
        vec![[0.2, 1.0], [7.7, 1.3], [1.4, 2.1], [4.0, 4.0]],
        vec![[0.0; 2]; 4],
        cell,
    )
    .unwrap();
    let result = evaluate(&state, ShiftedLennardJones::new(2.5).unwrap()).unwrap();
    let total = result.forces.iter().fold([0.0, 0.0], |sum, force| {
        [sum[0] + force[0], sum[1] + force[1]]
    });
    close(total[0], 0.0);
    close(total[1], 0.0);
}
