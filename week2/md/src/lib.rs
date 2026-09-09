/// Returns the program's greeting.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

/// Returns the Lennard-Jones pair energy at a separation in reduced units.
pub fn lennard_jones_pair_energy(distance: f64) -> f64 {
    let inverse_distance_sixth = distance.recip().powi(6);
    4.0 * (inverse_distance_sixth.powi(2) - inverse_distance_sixth)
}

/// Returns the radial Lennard-Jones pair force in reduced units.
pub fn lennard_jones_pair_force(_distance: f64) -> f64 {
    unimplemented!("Lennard-Jones pair force")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
