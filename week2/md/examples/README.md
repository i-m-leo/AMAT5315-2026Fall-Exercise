# Examples

From the repository root, generate the dimer energy-error plot with:

```bash
~/anaconda3/bin/python week2/md/examples/plot_dimer.py
```

The command runs the `dimer_energy` Rust example and writes `week2/dimer.png`.

## Lennard-Jones fluid CLI

From `week2/`, reproduce the default release-mode simulation with:

```bash
make reproduce
```

This is equivalent to:

```bash
/opt/homebrew/bin/cargo run --release --manifest-path md/Cargo.toml -- run --out artifacts
```

It writes `artifacts/run.json` and `artifacts/traj.jsonl`. Check the saved
trajectory by recomputing its physical quantities with:

```bash
/opt/homebrew/bin/cargo run --release --manifest-path md/Cargo.toml -- check --input artifacts
```

Render the 200 saved frames and their radial distribution function to video:

```bash
/opt/homebrew/bin/cargo run --release --manifest-path md/Cargo.toml -- video \
  --input artifacts --output artifacts/md.mp4
```

The `artifacts/` directory and Cargo's `md/target/` build directory are ignored
by Git.
