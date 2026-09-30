//! `run.json` and `result.json` writers.

use crate::experiment::Experiment;
use serde_json::{json, Map, Value};

/// `run.json`: the experiment file's path, and its fields except the two
/// grids, plus the recording schedule when it is on.
pub fn run_json(exp: &Experiment, experiment_file: &str, every: Option<usize>) -> Value {
    let mut fields = Map::new();
    for (key, value) in exp.raw.as_object().expect("experiment is an object") {
        if key != "background" && key != "perturbation" {
            fields.insert(key.clone(), value.clone());
        }
    }
    let mut root = Map::new();
    root.insert("experiment_file".into(), json!(experiment_file));
    root.insert("experiment".into(), Value::Object(fields));
    if let Some(every) = every {
        let steps: Vec<usize> = (every..=exp.steps).step_by(every).collect();
        let times: Vec<f64> = steps.iter().map(|&n| n as f64 * exp.dt).collect();
        root.insert(
            "recording".into(),
            json!({ "every": every, "steps": steps, "times": times }),
        );
    }
    Value::Object(root)
}

/// `result.json`: the run's mode and the grid, survey, and time-stepping
/// fields of the experiment.
pub fn result_json(exp: &Experiment, mode: &str) -> Value {
    json!({
        "mode": mode,
        "nx": exp.nx,
        "nz": exp.nz,
        "dx": exp.dx,
        "dt": exp.dt,
        "steps": exp.steps,
        "shots": exp.raw["shots"],
        "receivers": exp.raw["receivers"],
    })
}
