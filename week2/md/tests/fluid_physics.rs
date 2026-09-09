use md::fluid::{
    evaluate, evaluate_with_method, triangular_lattice, FluidConfig, FluidState, ForceMethod,
    PeriodicBox, ShiftedLennardJones,
};

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

fn assert_methods_agree(state: &FluidState) {
    let potential = ShiftedLennardJones::new(2.5).unwrap();
    let naive = evaluate_with_method(state, potential, ForceMethod::Naive).unwrap();
    let cells = evaluate_with_method(state, potential, ForceMethod::Cells).unwrap();
    assert!((naive.potential_energy - cells.potential_energy).abs() < 1.0e-12);
    for (left, right) in naive.forces.iter().zip(cells.forces.iter()) {
        assert!((left[0] - right[0]).abs() < 1.0e-11);
        assert!((left[1] - right[1]).abs() < 1.0e-11);
    }
}

#[test]
fn cell_list_matches_naive_on_a_perturbed_lattice() {
    let (cell, mut positions) = triangular_lattice(36, 0.8).unwrap();
    for (index, position) in positions.iter_mut().enumerate() {
        position[0] += 0.013 * (index as f64).sin();
        position[1] += 0.017 * (index as f64).cos();
        *position = cell.wrap(*position);
    }
    let state = FluidState::new(positions, vec![[0.0; 2]; 36], cell).unwrap();
    assert_methods_agree(&state);
}

#[test]
fn cell_list_matches_naive_for_boundary_and_cutoff_pairs() {
    let cell = PeriodicBox::new([10.0, 10.0]).unwrap();
    let positions = vec![
        [0.10, 1.0],
        [9.10, 1.0],
        [3.0, 3.0],
        [3.0 + 2.5 - 1.0e-10, 3.0],
        [6.0, 6.0],
        [6.0 + 2.5, 6.0],
    ];
    let state = FluidState::new(positions, vec![[0.0; 2]; 6], cell).unwrap();
    assert_methods_agree(&state);
}

#[test]
fn cell_list_matches_naive_in_a_two_cell_wide_box() {
    let cell = PeriodicBox::new([5.2, 5.2]).unwrap();
    let positions = vec![[0.2, 0.2], [2.7, 0.2], [0.2, 2.8], [3.0, 3.0]];
    let state = FluidState::new(positions, vec![[0.0; 2]; 4], cell).unwrap();
    assert_methods_agree(&state);
}

#[test]
fn cell_list_matches_naive_for_the_default_lattice() {
    let config = FluidConfig::default();
    let state = md::fluid::initialize_fluid(&config).unwrap();
    assert_methods_agree(&state);
}
