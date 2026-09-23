"""Compare the last Taylor-Green frame with the exact field, and draw the
vorticity at t = 0 and t = 1 with velocity arrows.

Reads week4/artifacts/taylor-green/{fields.jsonl,exact-t1.json}, prints the
relative velocity error of the last frame, and writes
week4/evidence/taylor-green.png. Run from week4/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "taylor-green"
FIELDS = ARTIFACTS / "fields.jsonl"
EXACT = ARTIFACTS / "exact-t1.json"
OUT = ROOT / "evidence" / "taylor-green.png"


def load_frames():
    with FIELDS.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relative_velocity_error(frame, exact):
    u = np.asarray(frame["u"], dtype=float)
    v = np.asarray(frame["v"], dtype=float)
    ue = np.asarray(exact["u"], dtype=float)
    ve = np.asarray(exact["v"], dtype=float)
    numerator = np.sum((u - ue) ** 2 + (v - ve) ** 2)
    denominator = np.sum(ue**2 + ve**2)
    return np.sqrt(numerator / denominator)


def main():
    frames = load_frames()
    with EXACT.open() as handle:
        exact = json.load(handle)

    first, last = frames[0], frames[-1]
    error = relative_velocity_error(last, exact)
    print(
        f"Taylor-Green velocity relative L2 error at t = {last['t']:.6f} "
        f"(step {last['step']}): {error:.6e}"
    )
    print(
        "  max |du| = "
        f"{max(abs(a - b) for a, b in zip(last['u'], exact['u'])):.3e}, "
        "max |dv| = "
        f"{max(abs(a - b) for a, b in zip(last['v'], exact['v'])):.3e}"
    )

    n = int(exact["n"])
    grid = np.arange(n) * 2.0 * np.pi / n
    xx, yy = np.meshgrid(grid, grid)
    stride = 4

    def omega(frame):
        return np.asarray(frame["omega"], dtype=float).reshape(n, n)

    omega0 = omega(first)
    omega1 = omega(last)
    limit = max(np.abs(omega0).max(), np.abs(omega1).max())

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, frame, values in ((axes[0], first, omega0), (axes[1], last, omega1)):
        mesh = ax.pcolormesh(
            xx, yy, values, shading="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, rasterized=True,
        )
        u = np.asarray(frame["u"], dtype=float).reshape(n, n)
        v = np.asarray(frame["v"], dtype=float).reshape(n, n)
        ax.quiver(
            xx[::stride, ::stride], yy[::stride, ::stride],
            u[::stride, ::stride], v[::stride, ::stride],
            color="black", alpha=0.6, pivot="mid",
        )
        ax.set_title(f"t = {frame['t']:.2f}")
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")

    fig.colorbar(mesh, ax=axes, label="vorticity $\\omega$", shrink=0.9)
    fig.suptitle(
        "Taylor-Green vorticity and velocity, n = 64, $\\nu$ = 0.1, "
        f"relative velocity error {error:.2e}"
    )
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
