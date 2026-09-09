# Lennard–Jones fluid CLI design

The Week 2 `md` crate provides `run`, `check`, and `video` subcommands for a
2D reduced-unit Lennard–Jones fluid. Runs use periodic minimum-image
coordinates, a triangular starting lattice, potential-shifted cutoff
interaction, seeded Gaussian velocities, velocity-Verlet integration, and
JSON/JSONL artifacts compatible with the Week 2 viewer.

The production force path is a periodic nine-cell list (`--force cells`), with
the original all-pairs evaluator retained as `--force naive`. A production
temperature ramp is optional (`--ramp-to`) and is recorded in `run.json`.
