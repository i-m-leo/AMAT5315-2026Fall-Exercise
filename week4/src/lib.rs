//! Time integrators and advection-diffusion rate functions for AMAT5315 week 4.

pub mod centred;
pub mod fourier;
pub mod grid;
pub mod integrators;
pub mod vorticity;

pub use centred::CentredDifferenceRhs;
pub use fourier::FourierRhs;
pub use grid::{gaussian, mode, wave_exact, PeriodicGrid};
pub use integrators::{ForwardEuler, Integrator, Midpoint, RungeKutta4, Scheme};
pub use vorticity::{Grid2d, Spectral};

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn l2_error(computed: &[f64], exact: &[f64]) -> f64 {
        let sum: f64 = computed
            .iter()
            .zip(exact)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        (sum / computed.len() as f64).sqrt()
    }

    #[test]
    fn wavenumbers_cover_positive_and_negative() {
        let grid = PeriodicGrid::new(8);
        assert_eq!(grid.wavenumbers(), vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn nyquist_mode_has_zero_spectral_derivative() {
        let grid = PeriodicGrid::new(8);
        let rhs = FourierRhs::new(grid, 1.0, 0.0);
        let state = mode(rhs.grid(), 4.0, 1.0, 0.0);
        let rate = rhs.rate(&state);
        let scale = rate.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(scale < 1e-12, "nyquist should not move, got max {scale}");
    }

    #[test]
    fn centred_nyquist_derivative_is_zero() {
        let grid = PeriodicGrid::new(8);
        let rhs = CentredDifferenceRhs::new(grid, 1.0, 0.0);
        let state = mode(rhs.grid(), 4.0, 1.0, 0.0);
        let rate = rhs.rate(&state);
        let scale = rate.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(scale < 1e-12, "centred nyquist should not move, got max {scale}");
    }

    #[test]
    fn fourier_matches_centred_on_smooth_mode() {
        let grid = PeriodicGrid::new(64);
        let c = 0.7;
        let nu = 0.05;
        let state = mode(&grid, 2.0, 1.0, 0.3);
        let spectral = FourierRhs::new(grid, c, nu).rate(&state);
        let finite = CentredDifferenceRhs::new(grid, c, nu).rate(&state);
        assert!(l2_error(&spectral, &finite) < 1e-2);
    }

    #[test]
    fn every_integrator_advances_a_single_wave() {
        let grid = PeriodicGrid::new(64);
        let c = 0.5;
        let nu = 0.0;
        let k = 1.0;
        let t_end = 1.0;
        let steps = 2000;
        let dt = t_end / steps as f64;
        let initial = mode(&grid, k, 1.0, 0.0);

        for scheme in Scheme::all() {
            let rhs = FourierRhs::new(grid, c, nu);
            let final_state = scheme.advance(&initial, dt, steps, |s| rhs.rate(s));
            let exact = wave_exact(&grid, k, 1.0, 0.0, c, nu, t_end);
            let error = l2_error(&final_state, &exact);
            assert!(
                error < 5e-3,
                "{} error too large: {error}",
                scheme.name()
            );
        }
    }

    #[test]
    fn rk4_beats_euler_on_a_single_wave() {
        let grid = PeriodicGrid::new(64);
        let rhs = FourierRhs::new(grid, 0.5, 0.01);
        let initial = mode(&grid, 1.0, 1.0, 0.0);
        let t_end = 1.0;
        let steps = 100;
        let dt = t_end / steps as f64;
        let exact = wave_exact(&grid, 1.0, 1.0, 0.0, 0.5, 0.01, t_end);

        let euler = ForwardEuler.advance(&initial, dt, steps, |s| rhs.rate(s));
        let rk4 = RungeKutta4.advance(&initial, dt, steps, |s| rhs.rate(s));
        assert!(l2_error(&rk4, &exact) < l2_error(&euler, &exact));
    }

    #[test]
    fn spectral_exact_matches_single_wave() {
        let grid = PeriodicGrid::new(64);
        let (c, nu) = (0.7, 0.05);
        let rhs = FourierRhs::new(grid, c, nu);
        let initial = mode(&grid, 3.0, 1.0, 0.4);
        for t in [0.0, 0.3, 1.7, 2.0 * PI] {
            let spectral = rhs.exact(&initial, t);
            let exact = wave_exact(&grid, 3.0, 1.0, 0.4, c, nu, t);
            assert!(
                l2_error(&spectral, &exact) < 1e-10,
                "t = {t}: {}",
                l2_error(&spectral, &exact)
            );
        }
    }

    #[test]
    fn gaussian_peak_is_at_its_centre() {
        let grid = PeriodicGrid::new(64);
        let values = gaussian(&grid, 0.25, PI / 2.0);
        let (argmax, peak) = values
            .iter()
            .enumerate()
            .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        assert!((peak - 1.0).abs() < 1e-12);
        assert!((grid.point(argmax) - PI / 2.0).abs() < grid.spacing());
    }

    #[test]
    fn single_wave_dispersion_matches_exact() {
        let grid = PeriodicGrid::new(16);
        let (c, nu, t_end) = (0.3, 0.02, 0.5);
        let exact = wave_exact(&grid, 1.0, 1.0, 0.0, c, nu, t_end);
        let wrapped = wave_exact(&grid, 1.0, 1.0, 2.0 * PI, c, nu, t_end);
        assert!((exact[0] - wrapped[0]).abs() < 1e-12);
        for (&value, &reference) in exact.iter().zip(&wrapped) {
            assert!((value - reference).abs() < 1e-12);
        }
    }
}
