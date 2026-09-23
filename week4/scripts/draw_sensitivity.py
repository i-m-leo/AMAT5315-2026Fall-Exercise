"""Draw the separation between the perturbed and unperturbed vorticity fields.

For each case the two RK4 runs differ only by the initial vorticity ripple
delta omega = -7e-5 * M * cos(3x) cos(4y). The plotted quantity is the relative
L2 distance

    d(t) = ||omega_perturbed(t) - omega_base(t)||_2 / ||omega_base(t)||_2

against time on a logarithmic distance axis.

Reads artifacts/sensitivity/*/fields.jsonl produced by
scripts/run_sensitivity.sh and writes evidence/sensitivity.png. Run from week4/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "artifacts" / "sensitivity"
OUT = ROOT / "evidence" / "sensitivity.png"

PAIRS = [
    ("tg-base", "tg-perturbed", "Taylor-Green, n = 64, $\\nu$ = 0.1", "#1f77b4"),
    ("rand-base", "rand-pert", "random, n = 128, $\\nu$ = 0.004", "#d62728"),
]


def load(name):
    times, omegas = [], []
    with (SCAN / name / "fields-full.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            times.append(frame["t"])
            omegas.append(np.asarray(frame["omega"], dtype=float))
    return np.asarray(times), np.asarray(omegas)


def main():
    fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)

    for base_name, pert_name, label, color in PAIRS:
        t_base, omega_base = load(base_name)
        t_pert, omega_pert = load(pert_name)
        assert t_base.shape == t_pert.shape and np.allclose(t_base, t_pert)

        difference = omega_pert - omega_base
        size = np.linalg.norm(omega_base.reshape(len(t_base), -1), axis=1)
        separation = np.linalg.norm(difference.reshape(len(t_base), -1), axis=1)
        relative = separation / size

        ax.semilogy(t_base, relative, color=color, linewidth=1.6, marker="o",
                    markersize=2.5, label=label)
        print(
            f"{label}: initial relative distance {relative[0]:.3e}, "
            f"final {relative[-1]:.3e}, growth factor {relative[-1]/relative[0]:.3e}"
        )

    ax.set_xlabel("time $t$")
    ax.set_ylabel("$\\|\\omega_{perturbed} - \\omega_{base}\\|_2 \\;/\\; \\|\\omega_{base}\\|_2$")
    ax.set_title("Growth of an initial vorticity ripple, RK4 dt = 0.01")
    ax.set_xlim(0.0, 20.0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
