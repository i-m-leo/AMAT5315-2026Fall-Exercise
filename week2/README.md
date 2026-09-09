# Week 2 molecular dynamics

The `md` crate contains the two-dimensional Lennard–Jones molecular-dynamics
simulation, trajectory checker, cell-list force evaluator, and trajectory video
renderer. Generated trajectories and build output are kept out of Git.

## Pages

[Open the Week 2 interactive trajectory viewer and diagnostics](https://i-m-leo.github.io/AMAT5315-2026Fall-Exercise/)

## Timing

Each row reports three runs from inside `week2/` on this machine. The NumPy
run used the Anaconda Python environment; Rust runs used the existing debug
and release binaries. The release Rust run wins, with a median below one third
of the debug median.

| Program | Median (s) | Range (s) |
| --- | ---: | ---: |
| NumPy `week2-sim.py` | 3.04 | 2.97 to 3.12 |
| debug `md run` | 6.86 | 6.77 to 7.00 |
| release `md run` | 0.51 | 0.51 to 0.52 |

Reproduce the measurements from `week2/` with:

```bash
~/anaconda3/bin/python week2-sim.py
md/target/debug/md run --out artifacts
md/target/release/md run --out artifacts
```
