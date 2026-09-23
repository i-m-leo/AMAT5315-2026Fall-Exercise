"""Relative omega error of the random-flow time-step series.

Compares the final vorticity of each RK4 run on the 128 x 128 random flow
against the dt = 0.0025 reference run, and fits the log-log slope of the error
series. Writes evidence/convergence.json.

Reads the runs written by scripts/run_convergence.sh. Run from week4/.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONV = ROOT / "artifacts" / "convergence"
OUT = ROOT / "evidence" / "convergence.json"

DTS = [0.02, 0.0125, 0.01]
REFERENCE_DT = 0.0025


def final_omega(dt):
    path = CONV / f"rk4-dt{dt}" / "fields-full.jsonl"
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return np.asarray(frame["omega"], dtype=float), frame["t"]


def main():
    reference, t_ref = final_omega(REFERENCE_DT)
    results = []
    for dt in DTS:
        omega, t = final_omega(dt)
        assert abs(t - t_ref) < 1e-12, (t, t_ref)
        error = np.linalg.norm(omega - reference) / np.linalg.norm(reference)
        results.append({"dt": dt, "steps": round(2.0 / dt), "relative_error": error})

    dts = np.array([entry["dt"] for entry in results])
    errors = np.array([entry["relative_error"] for entry in results])
    slope, intercept = np.polyfit(np.log(dts), np.log(errors), 1)

    document = {
        "case": "random",
        "n": 128,
        "seed": 2026,
        "k_band": [2, 6],
        "nu": 0.004,
        "t_end": 2.0,
        "method": "rk4",
        "quantity": "omega",
        "norm": "relative L2, ||omega - omega_ref||_2 / ||omega_ref||_2",
        "reference_dt": REFERENCE_DT,
        "runs": results,
        "loglog_slope": slope,
        "loglog_intercept": intercept,
    }
    with OUT.open("w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")

    print(f"reference: dt = {REFERENCE_DT}, t = {t_ref}")
    print(f"{'dt':>8} {'steps':>6} {'relative error':>16}")
    for entry in results:
        print(f"{entry['dt']:>8} {entry['steps']:>6} {entry['relative_error']:>16.6e}")
    print(f"log-log slope = {slope:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
