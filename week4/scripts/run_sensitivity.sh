#!/bin/sh
# Run each case twice with RK4 at dt = 0.01 to t = 20, snapshots every 0.5.
#
# Case 1: Taylor-Green, n = 64, nu = 0.1
# Case 2: random,      n = 128, nu = 0.004, seed 2026, wavenumbers 2..6
#
# The second run of each pair adds the vorticity ripple
#   delta omega = -7e-5 * M * cos(3x) cos(4y)
# with M the largest |u| or |v| of that case's initial field.
#
# Writes artifacts/sensitivity/<name>/{run.json,fields.jsonl} and
# artifacts/sensitivity/<name>.tsv. Run from week4/.
set -u

BIN=./target/release
OUT=artifacts/sensitivity
mkdir -p "$OUT"

run() {
    name=$1
    method=$2
    nu=$3
    dt=$4
    t_end=$5
    every=$6
    case_name=$7
    perturb=$8

    printf '%s\n' "-- $name"
    if [ "$case_name" = "taylor-green" ]; then
        base="$BIN/field taylor-green --n 64"
    else
        base="$BIN/field random --n 128 --seed 2026 --k-min 2 --k-max 6"
    fi

    if [ "$perturb" = "yes" ]; then
        sh -c "$base | $BIN/perturb" \
            | "$BIN/fluid" --method "$method" --nu "$nu" --dt "$dt" --t-end "$t_end" \
                --every "$every" --out "$OUT/$name" --full-precision > "$OUT/$name.tsv" 2> "$OUT/$name.err"
    else
        sh -c "$base" \
            | "$BIN/fluid" --method "$method" --nu "$nu" --dt "$dt" --t-end "$t_end" \
                --every "$every" --out "$OUT/$name" --full-precision > "$OUT/$name.tsv" 2> "$OUT/$name.err"
    fi
    code=$?
    last=$(tail -1 "$OUT/$name.tsv" | cut -f1)
    printf '   exit=%s  last_t=%s\n' "$code" "$last"
}

run tg-base      rk4 0.1   0.01 20 0.5 taylor-green no
run tg-perturbed rk4 0.1   0.01 20 0.5 taylor-green yes
run rand-base    rk4 0.004 0.01 20 0.5 random       no
run rand-pert    rk4 0.004 0.01 20 0.5 random       yes

printf '%s\n' "perturbation amplitudes:"
cat "$OUT"/*.err
