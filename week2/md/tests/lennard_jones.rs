use md::{lennard_jones_pair_energy, lennard_jones_pair_force};

const TOLERANCE: f64 = 1.0e-12;

#[test]
fn lennard_jones_pair_energy_in_reduced_units() {
    let energy = lennard_jones_pair_energy(2.0);
    let expected = -0.061_523_437_5;

    assert!((energy - expected).abs() < TOLERANCE);
}

#[test]
fn lennard_jones_pair_force_in_reduced_units() {
    let force = lennard_jones_pair_force(2.0);
    let expected = -0.181_640_625;

    assert!((force - expected).abs() < TOLERANCE);
}
