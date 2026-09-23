#!/bin/sh
# Run the week4 blow-up scan.
#
# Taylor-Green: n = 64, nu = 0.1, to t = 8, RK4 at dt = 0.032 and 0.033.
# Random:       n = 128, nu = 0.004, to t = 10, seed 2026, wavenumbers 2..6,
#               RK4 at dt = 0.040 and 0.038 (both specified), Euler at 0.01,
#               then RK4 at smaller steps until one reaches t = 10.
#
# Every run writes artifacts/scan/<name>/ (run.json, fields.jsonl) and
# artifacts/scan/<name>.tsv (t, E, Z up to the stop). Run from week4/.
set -u

BIN=./target/release
SCAN=artifacts/scan
mkdir -p "$SCAN"

run() {
    name=$1
    method=$2
    nu=$3
    dt=$4
    t_end=$5
    every=$6
    case_name=$7
    seed=$8

    printf '%s\n' "-- $name (method=$method nu=$nu dt=$dt t_end=$t_end)"
    if [ "$case_name" = "taylor-green" ]; then
        "$BIN/field" taylor-green --n 64
    else
        "$BIN/field" random --n 128 --seed "$seed" --k-min 2 --k-max 6
    fi | "$BIN/fluid" --method "$method" --nu "$nu" --dt "$dt" --t-end "$t_end" \
            --every "$every" --out "$SCAN/$name" > "$SCAN/$name.tsv" 2> "$SCAN/$name.err"
    code=$?
    last=$(tail -1 "$SCAN/$name.tsv" | cut -f1)
    printf '   exit=%s  last_t=%s\n' "$code" "$last"
}

# Largest speed of the random initial field.
"$BIN/field" random --n 128 --seed 2026 --k-min 2 --k-max 6 | python3 -c '
import json, math, sys
data = json.load(sys.stdin)
speed = max(math.hypot(a, b) for a, b in zip(data["u"], data["v"]))
print(f"largest speed of the random initial field: {speed:.6f}")
'

# Taylor-Green.
run tg-rk4-dt0.032  rk4 0.1 0.032 8 0.5 taylor-green 0
run tg-rk4-dt0.033  rk4 0.1 0.033 8 0.5 taylor-green 0

# Random, the two specified RK4 steps and the specified Euler step.
run rand-rk4-dt0.040  rk4   0.004 0.040 10 0.5 random 2026
run rand-rk4-dt0.038  rk4   0.004 0.038 10 0.5 random 2026
run rand-euler-dt0.01 euler 0.004 0.010 10 0.5 random 2026

# Smaller RK4 steps until one reaches t = 10.
run rand-rk4-dt0.036  rk4 0.004 0.036 10 0.5 random 2026
run rand-rk4-dt0.034  rk4 0.004 0.034 10 0.5 random 2026
run rand-rk4-dt0.033  rk4 0.004 0.033 10 0.5 random 2026
