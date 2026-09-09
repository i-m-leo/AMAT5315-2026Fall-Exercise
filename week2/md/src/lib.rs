/// Returns the program's greeting.
pub fn greeting() -> &'static str {
    "Hello, world!"
}

pub mod fluid;
mod simulation;

pub use simulation::{
    run_dimer_experiment, DimerState, EnergySample, ForwardEuler, Integrator, TimeIntegrator,
    VelocityVerlet,
};

/// Returns the Lennard-Jones pair energy at a separation in reduced units.
#[inline(never)]
pub fn lennard_jones_pair_energy(distance: f64) -> f64 {
    let inverse_distance_sixth = distance.recip().powi(6);
    4.0 * (inverse_distance_sixth.powi(2) - inverse_distance_sixth)
}

/// Returns the radial Lennard-Jones pair force in reduced units.
pub fn lennard_jones_pair_force(distance: f64) -> f64 {
    let inverse_distance = distance.recip();
    let inverse_distance_sixth = inverse_distance.powi(6);
    24.0 * inverse_distance * (2.0 * inverse_distance_sixth.powi(2) - inverse_distance_sixth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
