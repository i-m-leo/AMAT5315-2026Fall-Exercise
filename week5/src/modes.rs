//! Born scattered data and the adjoint (RTM) image.
//!
//! Both modes differentiate the same timestep that `src/kernel.rs` exports,
//! through Enzyme:
//!
//! * `born` holds the source, damping, and initial state fixed, and chains
//!   forward-mode (JVP) records of the timestep to propagate the experiment's
//!   velocity perturbation `perturbation` into the receivers.  The primal
//!   wavefield uses the background speed only.
//! * `adjoint` keeps the full trajectory, then chains reverse-mode (VJP)
//!   records of the timestep backward from `steps - 1` to `0`,
//!   back-propagating the receiver weights (`--data`, the Born data) and
//!   summing the velocity sensitivity over time and shots into the image.
//!
//! No hand-written tangent or adjoint is used; the wrappers in `crate::enzyme`
//! are the only derivatives.

use crate::enzyme::{step_jvp, step_vjp};
use crate::experiment::Experiment;
use crate::solver::{ricker, Model};

/// Where the adjoint keeps the trajectory. Only full history is implemented.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Full,
}

/// Per-shot counters of a reverse run.
pub struct ShotStats {
    pub shot: usize,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
}

/// The statistics object `[writes.reverse]` adds to `result.json`.
pub struct AdjointStats {
    pub storage: Storage,
    pub checkpoints: Option<usize>,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub saved_state_bytes: usize,
    pub state_wavefields: usize,
    pub per_shot: Vec<ShotStats>,
}

/// A shot's precomputed model, source footprint, and time-independent terms.
struct Setup<'a> {
    exp: &'a Experiment,
    model: Model,
    footprint: Vec<f64>,
}

impl<'a> Setup<'a> {
    fn new(exp: &'a Experiment, model: Model, shot: (f64, f64)) -> Setup<'a> {
        let (sx, sz) = shot;
        let footprint = (0..exp.nz)
            .flat_map(|iz| {
                (0..exp.nx).map(move |ix| {
                    let (dx, dz) = (ix as f64 - sx, iz as f64 - sz);
                    (-(dx * dx + dz * dz) / 2.0).exp()
                })
            })
            .collect();
        Setup {
            exp,
            model,
            footprint,
        }
    }

    /// The source at step `n`, already multiplied by the footprint.
    fn source(&self, n: usize) -> Vec<f64> {
        let amplitude = self.exp.amp * ricker(self.exp, n as f64 * self.exp.dt);
        self.footprint.iter().map(|g| amplitude * g).collect()
    }

    /// The background wave speed per cell.
    fn speed(&self) -> Vec<f64> {
        self.exp
            .background
            .iter()
            .zip(&self.exp.perturbation)
            .map(|(&c, _)| c)
            .collect()
    }

    fn step(&self, prev: &[f64], cur: &[f64], q: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.exp.nx * self.exp.nz];
        crate::enzyme::step_reference(
            prev,
            cur,
            &self.speed(),
            self.model.sigma(),
            q,
            self.exp.nx,
            self.exp.nz,
            self.exp.dt,
            self.model.inv_dx2(),
            &mut out,
        );
        out
    }

    /// `u^0 .. u^{steps+1}`; `u^0 = u^1 = 0`, and `u^{m+2}` is the step
    /// driven by the source at step `m`.
    fn trajectory(&self) -> Vec<Vec<f64>> {
        let n = self.exp.nx * self.exp.nz;
        let mut states = vec![vec![0.0; n], vec![0.0; n]];
        for m in 0..self.exp.steps {
            let q = self.source(m);
            let next = self.step(&states[m], &states[m + 1], &q);
            states.push(next);
        }
        states
    }
}

/// Born scattered data for every shot: `[shot][step][receiver]`, sampled
/// after each update like the forward traces.
pub fn born(exp: &Experiment) -> Result<Vec<f64>, String> {
    let n = exp.nx * exp.nz;
    let nr = exp.receivers.len();
    let mut data = vec![0.0; exp.shots.len() * exp.steps * nr];

    for (s, &shot) in exp.shots.iter().enumerate() {
        let setup = Setup::new(exp, Model::new(exp, false), shot);
        let primal = setup.trajectory();
        let speed = setup.speed();

        // Chain the timestep JVPs. The initial state and the source are
        // fixed, so only `c` carries a tangent: the perturbation.
        let mut prev = vec![0.0; n];
        let mut cur = vec![0.0; n];
        for m in 0..exp.steps {
            let q = setup.source(m);
            let tangent = step_jvp(
                &primal[m],
                &prev,
                &primal[m + 1],
                &cur,
                &speed,
                &exp.perturbation,
                setup.model.sigma(),
                &q,
                exp.nx,
                exp.nz,
                exp.dt,
                setup.model.inv_dx2(),
            )?;
            for (r, &(rx, rz)) in exp.receivers.iter().enumerate() {
                data[(s * exp.steps + m) * nr + r] = tangent[rz * exp.nx + rx];
            }
            prev = cur;
            cur = tangent;
        }
    }
    Ok(data)
}

/// The migrated image `[z][x]`: the velocity sensitivity summed over time and
/// shots, with the receiver weights read from `data` (`[shot][step][receiver]`).
pub fn adjoint(
    exp: &Experiment,
    data: &[f64],
    storage: Storage,
    every: Option<usize>,
) -> Result<(Vec<f64>, AdjointStats, Vec<f32>), String> {
    let n = exp.nx * exp.nz;
    let nr = exp.receivers.len();
    if data.len() != exp.shots.len() * exp.steps * nr {
        return Err(format!(
            "--data has {} values, expected {} x {} x {} = {}",
            data.len(),
            exp.shots.len(),
            exp.steps,
            nr,
            exp.shots.len() * exp.steps * nr
        ));
    }
    let mut image = vec![0.0; n];
    let mut frames: Vec<f32> = Vec::new();
    let mut per_shot = Vec::with_capacity(exp.shots.len());

    for (s, &shot) in exp.shots.iter().enumerate() {
        let setup = Setup::new(exp, Model::new(exp, false), shot);
        let primal = setup.trajectory();
        let speed = setup.speed();

        // One adjoint per state: a[0] .. a[steps+1].
        let mut a: Vec<Vec<f64>> = vec![vec![0.0; n]; exp.steps + 2];
        for m in 0..exp.steps {
            let base = (s * exp.steps + m) * nr;
            for (r, &(rx, rz)) in exp.receivers.iter().enumerate() {
                a[m + 2][rz * exp.nx + rx] += data[base + r];
            }
        }

        let mut reverse_calls = 0;
        for m in (0..exp.steps).rev() {
            let q = setup.source(m);
            let (prev_adj, cur_adj, c_adj) = step_vjp(
                &primal[m],
                &primal[m + 1],
                &speed,
                setup.model.sigma(),
                &q,
                exp.nx,
                exp.nz,
                exp.dt,
                setup.model.inv_dx2(),
                &a[m + 2],
            )?;
            // Accumulate into the input adjoints of the step. `c` is the same
            // array at every step, so its adjoint sums into the image.
            for i in 0..n {
                a[m][i] += prev_adj[i];
                a[m + 1][i] += cur_adj[i];
                image[i] += c_adj[i];
            }
            reverse_calls += 1;

            // The adjoint wavefield of the first shot: the adjoint of u^n
            // after reversing step n, for n decreasing over the same
            // schedule as the forward recording (n % every == 0).
            if let (Some(every), 0) = (every, s) {
                if m % every == 0 {
                    frames.extend(a[m].iter().map(|&v| v as f32));
                }
            }
        }

        per_shot.push(ShotStats {
            shot: s,
            reverse_calls,
            scheduler_forward_calls: exp.steps,
            peak_saved_states: exp.steps + 1,
        });
    }

    let saved_state_bytes = 2 * n * std::mem::size_of::<f64>();
    let stats = AdjointStats {
        storage,
        checkpoints: match storage {
            Storage::Full => None,
        },
        reverse_calls: exp.steps * exp.shots.len(),
        scheduler_forward_calls: exp.steps * exp.shots.len(),
        peak_saved_states: exp.steps + 1,
        saved_state_bytes,
        state_wavefields: 2,
        per_shot,
    };
    Ok((image, stats, frames))
}
