#!/bin/sh
# Time-step order study for Taylor-Green on an 8 x 8 grid, nu = 0.5, to t = 2.
#
# RK4 at dt = 0.4, 0.25, and 0.2, each under artifacts/order/rk4-dt<dt>/.
# Also records the exact field at t = 2 for the error comparison.
# Full-precision snapshots are kept so the error is not limited by the
# 6-decimal rounding of the default fields.jsonl.
#
# Run from week4/.
set -u

BIN=./target/release
OUT=artifacts/order
mkdir -p "$OUT"

run() {
    dt=$1
    name="rk4-dt$dt"
    printf '%s\n' "-- $name"
    "$BIN/field" taylor-green --n 8 \
        | "$BIN/fluid" --method rk4 --nu 0.5 --dt "$dt" --t-end 2 --every 2 \
            --out "$OUT/$name" --full-precision > "$OUT/$name.tsv" 2> "$OUT/$name.err"
    printf '   exit=%s  frames=%s\n' "$?" "$(wc -l < "$OUT/$name/fields-full.jsonl")"
}

run 0.4
run 0.25
run 0.2

"$BIN/field" taylor-green --n 8 --nu 0.5 --t 2 > "$OUT/exact-t2.json"
printf '%s\n' "wrote $OUT/exact-t2.json"
