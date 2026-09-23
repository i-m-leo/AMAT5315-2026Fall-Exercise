//! Evolve a periodic Gaussian pulse one lap around the box three ways and
//! report the maximum error of each run against the exact periodic solution.
//!
//! Runs (n = 64, c = 1, nu = 0.002, until t = 2 pi):
//!   RK4 + Fourier derivatives,   dt = 0.02
//!   RK4 + centred differences,   dt = 0.02
//!   forward Euler + Fourier,     dt = 0.005
//!
//! Usage: accuracy_run [out_dir]
//! Writes `line-accuracy.txt` (errors) and `line-accuracy-profiles.txt`
//! (x, exact, and the three final profiles) into the given directory.

use advection::{
    gaussian, CentredDifferenceRhs, ForwardEuler, FourierRhs, Integrator, PeriodicGrid,
    RungeKutta4,
};
use std::f64::consts::PI;
use std::path::PathBuf;

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.002;
const SIGMA: f64 = 0.25;
const X0: f64 = PI / 2.0;
const T_END: f64 = 2.0 * PI;

fn max_error(computed: &[f64], exact: &[f64]) -> f64 {
    computed
        .iter()
        .zip(exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("evidence"));
    std::fs::create_dir_all(&out_dir).expect("create output directory");

    let grid = PeriodicGrid::new(N);
    let initial = gaussian(&grid, SIGMA, X0);
    let fourier = FourierRhs::new(grid, C, NU);
    let centred = CentredDifferenceRhs::new(grid, C, NU);

    struct Run {
        label: &'static str,
        steps: usize,
        dt: f64,
        values: Vec<f64>,
    }

    let mut runs = Vec::new();

    let dt_rk4 = 0.02;
    let steps_rk4 = (T_END / dt_rk4).round() as usize;
    runs.push(Run {
        label: "RK4 Fourier  dt=0.02",
        steps: steps_rk4,
        dt: dt_rk4,
        values: RungeKutta4.advance(&initial, dt_rk4, steps_rk4, |s| fourier.rate(s)),
    });
    runs.push(Run {
        label: "RK4 centred  dt=0.02",
        steps: steps_rk4,
        dt: dt_rk4,
        values: RungeKutta4.advance(&initial, dt_rk4, steps_rk4, |s| centred.rate(s)),
    });

    let dt_euler = 0.005;
    let steps_euler = (T_END / dt_euler).round() as usize;
    runs.push(Run {
        label: "Euler Fourier dt=0.005",
        steps: steps_euler,
        dt: dt_euler,
        values: ForwardEuler.advance(&initial, dt_euler, steps_euler, |s| fourier.rate(s)),
    });

    let mut report = String::new();
    report.push_str(&format!(
        "periodic Gaussian pulse: n={N}, c={C}, nu={NU}, sigma={SIGMA}, x0={X0:.6}, t=2pi\n"
    ));
    report.push_str(&format!(
        "{:<24} {:>6} {:>14} {:>14}\n",
        "run", "steps", "t_final", "max |u - exact|"
    ));

    let mut profiles = String::new();
    profiles.push_str("x exact");
    for run in &runs {
        profiles.push_str(&format!(" {}", run.label.replace(' ', "_")));
    }
    profiles.push('\n');

    let exact_final: Vec<Vec<f64>> = runs
        .iter()
        .map(|run| fourier.exact(&initial, run.steps as f64 * run.dt))
        .collect();
    let exact_box = fourier.exact(&initial, T_END);

    for (run, exact) in runs.iter().zip(&exact_final) {
        let t_final = run.steps as f64 * run.dt;
        report.push_str(&format!(
            "{:<24} {:>6} {:>14.8} {:>14.6e}\n",
            run.label,
            run.steps,
            t_final,
            max_error(&run.values, exact)
        ));
    }

    for j in 0..N {
        profiles.push_str(&format!("{:.10e}", grid.point(j)));
        profiles.push_str(&format!(" {:.10e}", exact_box[j]));
        for run in &runs {
            profiles.push_str(&format!(" {:.10e}", run.values[j]));
        }
        profiles.push('\n');
    }

    let report_path = out_dir.join("line-accuracy.txt");
    let profiles_path = out_dir.join("line-accuracy-profiles.txt");
    std::fs::write(&report_path, &report).expect("write report");
    std::fs::write(&profiles_path, &profiles).expect("write profiles");
    print!("{report}");
}
