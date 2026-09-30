#!/usr/bin/env python3
"""Check the born and adjoint runs against the course reference numbers.

The runs are produced with

    seismic --experiment inputs/reflector.json --mode born --out artifacts/born
    seismic --experiment inputs/reflector.json --mode adjoint \
            --data artifacts/born/born_data.npy --out artifacts/adjoint

Checks the Born data norm, the dot-product identity, and that the image's
depth profile peaks on the reflector, and writes artifacts/adjoint/verify.txt.
Run from week5/.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs" / "reflector.json"
BORN = ROOT / "artifacts" / "born"
ADJOINT = ROOT / "artifacts" / "adjoint"

REFERENCE_BORN_NORM = 0.18667589618149358
REFERENCE_DOT = 0.03484789021516377
TOL = 1e-12


def main():
    exp = json.loads(INPUTS.read_text())
    born = np.load(BORN / "born_data.npy")
    image = np.load(ADJOINT / "image.npy")
    perturbation = np.asarray(exp["perturbation"], dtype=float)
    result = json.loads((ADJOINT / "result.json").read_text())

    lines = []
    ok = True

    def check(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        lines.append(f"{name:18s} {'ok' if good else 'FAIL'}  {detail}")

    n_shots, steps, n_receivers = len(exp["shots"]), exp["steps"], len(exp["receivers"])
    check(
        "born shape",
        born.shape == (n_shots, steps, n_receivers) and born.dtype == np.float64,
        f"born_data {born.shape}/{born.dtype}",
    )
    check(
        "image shape",
        image.shape == (exp["nz"], exp["nx"]) and image.dtype == np.float64,
        f"image {image.shape}/{image.dtype}",
    )

    norm = float(np.sqrt((born**2).sum()))
    check(
        "born norm",
        abs(norm - REFERENCE_BORN_NORM) < TOL * REFERENCE_BORN_NORM,
        f"{norm!r} vs reference {REFERENCE_BORN_NORM!r}",
    )

    dot_left = float((born**2).sum())
    dot_right = float((image.ravel() * perturbation.ravel()).sum())
    relative = abs(dot_left - dot_right) / abs(dot_left)
    check(
        "dot product",
        relative < 1e-12,
        f"<Born, Born> = {dot_left!r}, <image, dv> = {dot_right!r}, relative {relative:.3e}",
    )
    check(
        "dot reference",
        abs(dot_left - REFERENCE_DOT) < TOL * REFERENCE_DOT,
        f"{dot_left!r} vs reference {REFERENCE_DOT!r}",
    )

    profile = np.abs(image).sum(axis=1)
    peak = int(profile.argmax())
    cell_km = exp["dx"] * exp["length_unit_m"] / 1000.0
    check(
        "reflector depth",
        peak == 21,
        f"depth profile peaks at z = {peak} ({peak * cell_km:.1f} km), reflector at z = 21",
    )
    check(
        "image peak",
        float(np.abs(image).max()) > 0.0,
        f"peak |image| {np.abs(image).max():.4e}",
    )

    stats = result["statistics"]
    check(
        "statistics",
        stats["storage"] == "full"
        and stats["checkpoints"] is None
        and stats["reverse_calls"] == steps * n_shots
        and stats["scheduler_forward_calls"] == steps * n_shots
        and stats["peak_saved_states"] == steps + 1
        and len(stats["per_shot"]) == n_shots,
        f"storage {stats['storage']}, {stats['reverse_calls']} reverse calls, "
        f"peak {stats['peak_saved_states']} states ({stats['peak_saved_bytes']} bytes)",
    )

    report = "\n".join(lines) + "\n"
    (ADJOINT / "verify.txt").write_text(report)
    print(report, end="")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
