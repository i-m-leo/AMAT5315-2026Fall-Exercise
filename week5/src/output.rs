//! `run.json` and `result.json` writers.

use crate::experiment::Experiment;
use serde_json::{json, Map, Value};

/// `run.json`: the experiment file's path, and its fields except the two
/// grids, plus the recording schedule when it is on. Forward records
/// `every, 2*every, ..., steps`; the adjoint records the reversed schedule
/// `steps-1, ..., every, 0` in frame order.
pub fn run_json(
    exp: &Experiment,
    experiment_file: &str,
    every: Option<usize>,
    adjoint: bool,
) -> Value {
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
        let steps: Vec<usize> = if adjoint {
            (0..exp.steps)
                .rev()
                .filter(|n| n % every == 0)
                .collect()
        } else {
            (every..=exp.steps).step_by(every).collect()
        };
        let times: Vec<f64> = steps.iter().map(|&n| n as f64 * exp.dt).collect();
        root.insert(
            "recording".into(),
            json!({ "every": every, "steps": steps, "times": times }),
        );
    }
    Value::Object(root)
}

/// `result.json`: the run's mode and the grid, survey, and time-stepping
/// fields of the experiment. The adjoint adds its statistics object.
pub fn result_json(exp: &Experiment, mode: &str) -> Value {
    result_json_with(exp, mode, None)
}

/// `result.json` with the optional reverse-mode statistics.
pub fn result_json_with(
    exp: &Experiment,
    mode: &str,
    statistics: Option<crate::modes::AdjointStats>,
) -> Value {
    let mut fields = Map::new();
    fields.insert("mode".into(), json!(mode));
    fields.insert("nx".into(), json!(exp.nx));
    fields.insert("nz".into(), json!(exp.nz));
    fields.insert("dx".into(), json!(exp.dx));
    fields.insert("dt".into(), json!(exp.dt));
    fields.insert("steps".into(), json!(exp.steps));
    fields.insert("shots".into(), exp.raw["shots"].clone());
    fields.insert("receivers".into(), exp.raw["receivers"].clone());
    if let Some(stats) = statistics {
        let per_shot: Vec<Value> = stats
            .per_shot
            .iter()
            .map(|s| {
                json!({
                    "shot": s.shot,
                    "reverse_calls": s.reverse_calls,
                    "scheduler_forward_calls": s.scheduler_forward_calls,
                    "peak_saved_states": s.peak_saved_states,
                })
            })
            .collect();
        fields.insert(
            "statistics".into(),
            json!({
                "storage": match stats.storage {
                    crate::modes::Storage::Full => "full",
                },
                "checkpoints": stats.checkpoints,
                "reverse_calls": stats.reverse_calls,
                "scheduler_forward_calls": stats.scheduler_forward_calls,
                "peak_saved_states": stats.peak_saved_states,
                "peak_saved_bytes": stats.peak_saved_states * stats.saved_state_bytes,
                "saved_state_bytes": stats.saved_state_bytes,
                "state_wavefields": stats.state_wavefields,
                "per_shot": per_shot,
            }),
        );
    }
    Value::Object(fields)
}
