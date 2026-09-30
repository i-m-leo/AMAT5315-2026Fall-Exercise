#!/usr/bin/env python3
"""Check a forward run against the course reference numbers.

Runs are produced with

    cargo run --release -- --experiment inputs/reflector.json \
        --mode forward --every 3 --out artifacts/forward

Compares artifacts/forward with the reference forward run: per-shot L2 norms,
trace norm, trace peaks, the recording schedule, and the physics of the echo.
Writes artifacts/forward/verify.txt. Run from week5/.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "forward"
TOL = 1e-9

REFERENCE = {
    "per_shot_l2": [6.623960807913, 6.798645121807, 6.623960807913],
    "trace_norm": 11.574769503614014,
    "peaks": [  # shot, trace index, receiver, |pressure|
        (0, 83, 1, 0.60809514342744),
        (1, 83, 6, 0.5927139734671293),
        (2, 83, 12, 0.60809514342744),
    ],
}


def main():
    run = json.loads((OUT / "run.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    traces = np.load(OUT / "traces.npy")
    wavefield = np.load(OUT / "wavefield.npy")
    echo = np.load(OUT / "echo.npy")

    lines = []
    ok = True

    def check(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"{name:12s} {'ok' if good else 'FAIL'}  {detail}")

    check(
        "shapes",
        traces.shape == (3, 240, 14)
        and traces.dtype == np.float64
        and wavefield.shape == echo.shape == (80, 41, 41)
        and wavefield.dtype == echo.dtype == np.float32,
        f"traces {traces.shape}/{traces.dtype}, wavefield {wavefield.shape}/{wavefield.dtype}",
    )
    check("mode", result["mode"] == "forward", f"result.json mode {result['mode']}")

    per_shot = np.sqrt((traces**2).sum(axis=(1, 2)))
    scale = max(REFERENCE["per_shot_l2"])
    check(
        "per-shot l2",
        np.abs(per_shot - REFERENCE["per_shot_l2"]).max() < TOL * scale,
        f"{per_shot[0]:.12e} {per_shot[1]:.12e} {per_shot[2]:.12e}",
    )
    norm = float(np.sqrt((traces**2).sum()))
    check(
        "trace norm",
        abs(norm - REFERENCE["trace_norm"]) < TOL * REFERENCE["trace_norm"],
        f"{norm!r} vs {REFERENCE['trace_norm']!r}",
    )

    flat = np.abs(traces.reshape(3, -1))
    peaks = [(s, flat[s].argmax() // 14, flat[s].argmax() % 14, flat[s].max()) for s in range(3)]
    peaks_ok = all(
        (s, trace, recv) == want[:3] and abs(amp - want[3]) < TOL * abs(want[3])
        for (s, trace, recv, amp), want in zip(peaks, REFERENCE["peaks"])
    )
    check(
        "trace peaks",
        peaks_ok,
        "; ".join(f"shot {s} trace {t} recv {r} |p| {a:.12e}" for s, t, r, a in peaks),
    )

    # Recording schedule: frames at every, 2*every, ..., and times as
    # float64 products of step and dt (not recomputed from a single total).
    recording = run["recording"]
    steps = recording["steps"]
    times = [s * result["dt"] for s in steps]
    check(
        "recording",
        recording["every"] == 3
        and steps == list(range(3, 241, 3))
        and times == recording["times"]
        and len(steps) == wavefield.shape[0],
        f"{len(steps)} frames, steps {steps[0]}..{steps[-1]}, "
        f"times {times[0]!r}..{times[-1]!r} = step*dt",
    )

    # Echo physics: the perturbation reflects at z = 21. The first strong
    # echo frame must peak on that row, and its distance from the shot must
    # be the half-way distance c t / 2.
    depth, shot_x, shot_z, c = 21, 10, 8, 1.8
    frames = [k for k in range(len(echo)) if np.abs(echo[k]).max() > 1e-3]
    k = frames[0]
    iz, ix = np.unravel_index(np.abs(echo[k]).argmax(), echo[k].shape)
    radius = float(np.hypot(ix - shot_x, iz - shot_z))
    travel = c * steps[k] * result["dt"] / 2
    check(
        "echo reflector",
        iz == depth and abs(radius - travel) <= 1.5,
        f"frame {k} (step {steps[k]}, t={times[k]}) peaks at ({ix}, {iz}), "
        f"r = {radius:.2f} cells vs c t/2 = {travel:.2f}",
    )

    # The wavefield is the background model and must be free of the echo.
    check(
        "wavefield bg",
        np.abs(wavefield).max() > 0
        and not np.array_equal(wavefield, echo),
        f"peak |wavefield| {np.abs(wavefield).max():.3e}",
    )

    report = "\n".join(lines) + "\n"
    (OUT / "verify.txt").write_text(report)
    print(report, end="")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
