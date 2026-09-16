# Week 3 Ising evidence

This guide regenerates the committed Week 3 code outputs and evidence from a clean
clone. Run all commands from `week3/`.

## Install

```sh
cargo install --path . --quiet
```

The plotting scripts use the `ising-plot` conda environment:

```sh
conda activate ising-plot
```

## Record the simulations

The four Metropolis runs are:

```sh
ising --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
ising --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
ising --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

The two Wolff runs are:

```sh
ising --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
ising --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
```

The contract ramp also regenerates the committed recording input. The command
writes `runs/ramp/spins.jsonl`; copy it to the committed top-level recording:

```sh
ising --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
cp runs/ramp/spins.jsonl spins.jsonl
```

## Evidence, in production order

Each command below is run from `week3/`. The listed files are the files it writes.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/peaks.py
```

Writes: `evidence/peaks.txt`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/errors.py
```

Writes: `evidence/errors.txt`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python evidence/draw_boltzmann.py
```

Writes: `evidence/boltzmann.png`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python evidence/draw_ising_observables.py
```

Writes: `evidence/magnetization.png` and `evidence/susceptibility.png`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/trace.py
```

Writes: `evidence/trace.png`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/acf_binning.py
```

Writes: `evidence/acf-binning.png`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/tau.py
```

Writes: `evidence/tau.png`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/bootstrap.py
```

Writes: `evidence/chi-bootstrap.png` and `evidence/chi-bootstrap.txt`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/magnetization_compare.py
```

Writes: `evidence/magnetization-compare.png`.

```sh
MPLCONFIGDIR=/tmp/ising-mplconfig python scripts/compare.py
```

Writes: `evidence/tau-compare.png`.

The command-record files are documentation of the drawing commands:

```text
evidence/DRAW_COMMAND.txt                  -> draw_boltzmann.py / boltzmann.png
evidence/TRACE_COMMAND.txt                -> trace.py / trace.png
evidence/ACF_BINNING_COMMAND.txt          -> acf_binning.py / acf-binning.png
evidence/TAU_COMMAND.txt                  -> tau.py / tau.png
evidence/MAGNETIZATION_COMPARE_COMMAND.txt -> magnetization_compare.py / magnetization-compare.png
```

All simulation artifacts under `artifacts/` are local inputs and are intentionally
not tracked. Sampling uncertainty remains largest for Metropolis near the critical
temperature because its autocorrelation time is long; the block-bootstrap and
block-length diagnostics make that limitation visible in the evidence.
