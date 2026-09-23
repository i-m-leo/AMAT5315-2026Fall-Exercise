#!/bin/sh
# Time-step convergence of the random flow on a 128 x 128 grid.
#
# Random case: n = 128, nu = 0.004, seed 2026, wavenumbers 2..6, to t = 2.
# RK4 at dt = 0.02, 0.0125, and 0.01, compared with a dt = 0.0025 reference.
# The final field of every run is retained (full precision) for the error
# estimate. Run from week4/.
set -u

BIN=./target/release
OUT=artifacts/convergence
mkdir -p "$OUT"

run() {
    dt=$1
    name="rk4-dt$dt"
    printf '%s\n' "-- $name"
    "$BIN/field" random --n 128 --seed 2026 --k-min 2 --k-max 6 \
        | "$BIN/fluid" --method rk4 --nu 0.004 --dt "$dt" --t-end 2 --every 2 \
            --out "$OUT/$name" --full-precision > "$OUT/$name.tsv" 2> "$OUT/$name.err"
    printf '   exit=%s  frames=%s\n' "$?" "$(wc -l < "$OUT/$name/fields-full.jsonl")"
}

run 0.02
run 0.0125
run 0.01
run 0.0025
