//! Parsing and validation of a `week5-seismic-experiment-v1` JSON file.

use serde_json::Value;

/// A parsed experiment. The raw JSON is kept so `run.json` and `result.json`
/// can echo the original fields and number types.
pub struct Experiment {
    pub raw: Value,
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub f0: f64,
    pub t0: f64,
    pub amp: f64,
    pub shots: Vec<(f64, f64)>,
    pub receivers: Vec<(usize, usize)>,
    pub sponge_width: f64,
    pub sponge_strength: f64,
    pub background: Vec<f64>,
    pub perturbation: Vec<f64>,
}

fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("field `{key}` is missing or not a number"))
}

fn count(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| format!("field `{key}` is missing or not a non-negative integer"))
}

fn pairs(v: &Value, key: &str) -> Result<Vec<(f64, f64)>, String> {
    let items = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("field `{key}` is missing or not an array"))?;
    items
        .iter()
        .map(|item| {
            let pair = item
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| format!("`{key}` entries must be [x, z] pairs"))?;
            let x = pair[0]
                .as_f64()
                .ok_or_else(|| format!("`{key}` coordinates must be numbers"))?;
            let z = pair[1]
                .as_f64()
                .ok_or_else(|| format!("`{key}` coordinates must be numbers"))?;
            Ok((x, z))
        })
        .collect()
}

fn grid(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let rows = v
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("field `{key}` is missing or not an array"))?;
    let mut flat = Vec::with_capacity(rows.len() * n);
    for row in rows {
        let row = row
            .as_array()
            .ok_or_else(|| format!("`{key}` must be a [z][x] array"))?;
        for value in row {
            flat.push(
                value
                    .as_f64()
                    .ok_or_else(|| format!("`{key}` must hold numbers"))?,
            );
        }
    }
    Ok(flat)
}

impl Experiment {
    /// Read, parse, and validate an experiment file.
    pub fn load(path: &std::path::Path) -> Result<Experiment, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let raw: Value =
            serde_json::from_str(&text).map_err(|e| format!("cannot parse {}: {e}", path.display()))?;
        Experiment::from_value(raw)
    }

    /// Validate an already-parsed experiment object.
    pub fn from_value(raw: Value) -> Result<Experiment, String> {
        let nx = count(&raw, "nx")?;
        let nz = count(&raw, "nz")?;
        if nx < 3 || nz < 3 {
            return Err(format!("grid must be at least 3 x 3, got {nx} x {nz}"));
        }
        let n = nx * nz;
        let dx = number(&raw, "dx")?;
        let dt = number(&raw, "dt")?;
        if dx <= 0.0 || dt <= 0.0 {
            return Err("dx and dt must be positive".into());
        }
        let steps = count(&raw, "steps")?;
        if steps < 1 {
            return Err("steps must be at least 1".into());
        }

        let shots = pairs(&raw, "shots")?;
        if shots.is_empty() {
            return Err("shots must not be empty".into());
        }
        for &(sx, sz) in &shots {
            if !(0.0..=nx as f64 - 1.0).contains(&sx) || !(0.0..=nz as f64 - 1.0).contains(&sz) {
                return Err(format!("shot ({sx}, {sz}) lies off the grid"));
            }
        }
        let receiver_pairs = pairs(&raw, "receivers")?;
        if receiver_pairs.is_empty() {
            return Err("receivers must not be empty".into());
        }
        let mut receivers = Vec::with_capacity(receiver_pairs.len());
        for &(rx, rz) in &receiver_pairs {
            if rx < 0.0 || rz < 0.0 || rx.fract() != 0.0 || rz.fract() != 0.0 {
                return Err(format!("receiver ({rx}, {rz}) must be a grid index"));
            }
            let (ix, iz) = (rx as usize, rz as usize);
            if ix >= nx || iz >= nz {
                return Err(format!("receiver ({ix}, {iz}) lies off the grid"));
            }
            receivers.push((ix, iz));
        }

        let background = grid(&raw, "background", nx)?;
        if background.len() != n {
            return Err(format!(
                "background has {} values, expected {n}",
                background.len()
            ));
        }
        let perturbation = grid(&raw, "perturbation", nx)?;
        if perturbation.len() != n {
            return Err(format!(
                "perturbation has {} values, expected {n}",
                perturbation.len()
            ));
        }

        Ok(Experiment {
            f0: number(&raw, "source_frequency")?,
            t0: number(&raw, "source_peak_time")?,
            amp: number(&raw, "source_amplitude")?,
            sponge_width: number(&raw, "sponge_width")?,
            sponge_strength: number(&raw, "sponge_strength")?,
            raw,
            nx,
            nz,
            dx,
            dt,
            steps,
            shots,
            receivers,
            background,
            perturbation,
        })
    }
}
