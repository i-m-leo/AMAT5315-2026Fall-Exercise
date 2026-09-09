# Week 2 review findings

## Findings and disposition

| Finding | Disposition |
| --- | --- |
| Force evaluation needed an unambiguous reference path. | Fixed: `--force naive` retains the all-pairs evaluator; `--force cells` is the default. |
| Cell-list boundary aliases could double-count or miss pairs. | Fixed: wrapped neighbor cells are deduplicated, pairs use `j > i`, and tests cover boundary, cutoff, perturbed-lattice, and two-cell-wide cases. |
| Long runs can diverge from the reference through floating-point accumulation order. | Fixed: cell candidates are sorted by particle index before accumulation. |
| Stored energies must not be trusted by `check`. | Fixed: `check` recomputes potential and kinetic energies from positions and velocities. |
| Heating runs should remain distinguishable from NVE production runs. | Fixed: `ramp_to` is optional and recorded in `run.json`; the README documents the heating evidence. |
| The generated trajectory/video artifacts are large binary evidence. | Accepted: they are committed as assignment evidence; build output and temporary profiler data remain ignored. |

## Code ownership notes

`FluidState` owns the particle `Vec` arrays (`positions`, `velocities`, and
`forces`). Force evaluation borrows the state through `&FluidState`, and the
integrator mutates it through `&mut FluidState`. The generic `Integrator<S>`
trait lets the dimer experiment run with either Forward Euler or
velocity-Verlet through the same driver. The `#[derive(...)]` attributes are a
representative macro use: they generate parsing, serialization, and value-type
implementations at compile time.
