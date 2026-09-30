//! Seismic modelling: forward acoustic wave simulation and (later) its Born
//! and adjoint companions, for AMAT5315 week 5.

pub mod enzyme;
pub mod experiment;
pub mod modes;
pub mod npy;
pub mod output;
pub mod solver;

pub use experiment::Experiment;
pub use modes::{adjoint, born, AdjointStats, Storage};
pub use solver::{ricker, Model, Solver};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ricker_peaks_at_t0() {
        let exp = Experiment::from_value(serde_json::json!({
            "nx": 3, "nz": 3, "dx": 1.0, "dt": 0.1, "steps": 1,
            "source_frequency": 0.08, "source_peak_time": 15.0, "source_amplitude": 1.0,
            "shots": [[1.0, 1.0]], "receivers": [[1, 1]],
            "sponge_width": 1.0, "sponge_strength": 0.0,
            "background": vec![vec![1.0; 3]; 3], "perturbation": vec![vec![0.0; 3]; 3],
        }))
        .unwrap();
        assert!((ricker(&exp, exp.t0) - 1.0).abs() < 1e-12);
        assert!(ricker(&exp, 0.0).abs() < 2e-5);
    }

    #[test]
    fn sponge_is_zero_in_the_interior() {
        let exp = Experiment::from_value(serde_json::json!({
            "nx": 9, "nz": 9, "dx": 1.0, "dt": 0.1, "steps": 1,
            "source_frequency": 0.08, "source_peak_time": 15.0, "source_amplitude": 1.0,
            "shots": [[4.0, 4.0]], "receivers": [[4, 4]],
            "sponge_width": 2.0, "sponge_strength": 0.4,
            "background": vec![vec![1.0; 9]; 9], "perturbation": vec![vec![0.0; 9]; 9],
        }))
        .unwrap();
        let model = Model::new(&exp, false);
        assert_eq!(model.sigma_at(4, 4), 0.0);
        assert!((model.sigma_at(0, 0) - 0.4).abs() < 1e-12);
        assert!((model.sigma_at(1, 4) - 0.4 * 0.25).abs() < 1e-12);
    }
}
