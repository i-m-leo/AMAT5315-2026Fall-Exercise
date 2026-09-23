"""Convergence figure with a fourth-order Richardson step choice.

Reads the measured errors in evidence/convergence.json and the retained fields
under artifacts/convergence/, then:

* plots the measured relative omega error against the time step on log-log axes
  with a fitted slope labelled in the legend;
* estimates the relative temporal error at dt = 0.01 by fourth-order Richardson
  extrapolation of the dt = 0.02 and 0.01 fields;
* predicts the error at the candidate steps 0.02, 0.0125, and 0.01 by
  fourth-order scaling from that estimate;
* picks the largest candidate whose predicted error is below 5e-6 and marks it
  on the plot.

Writes evidence/convergence.png. Run from week4/.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONV = ROOT / "artifacts" / "convergence"
REPORT = ROOT / "evidence" / "convergence.json"
OUT = ROOT / "evidence" / "convergence.png"

THRESHOLD = 5e-6
ORDER = 4
CANDIDATES = [0.02, 0.0125, 0.01]
RICHARDSON_FINE = 0.01
RICHARDSON_COARSE = 0.02


def final_omega(dt):
    path = CONV / f"rk4-dt{dt}" / "fields-full.jsonl"
    frame = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frame = json.loads(line)
    return np.asarray(frame["omega"], dtype=float)


def main():
    with REPORT.open() as handle:
        report = json.load(handle)
    measured_dt = np.array([run["dt"] for run in report["runs"]])
    measured_error = np.array([run["relative_error"] for run in report["runs"]])
    slope, intercept = np.polyfit(np.log(measured_dt), np.log(measured_error), 1)

    # Fourth-order Richardson estimate of the error in the finer field.
    omega_fine = final_omega(RICHARDSON_FINE)
    omega_coarse = final_omega(RICHARDSON_COARSE)
    scale = np.linalg.norm(omega_fine)
    ratio = RICHARDSON_FINE / RICHARDSON_COARSE
    richardson = (
        np.linalg.norm(omega_fine - omega_coarse)
        / (ratio ** -ORDER - 1.0)
        / scale
    )

    predicted = {
        dt: richardson * (dt / RICHARDSON_FINE) ** ORDER for dt in CANDIDATES
    }
    eligible = [dt for dt in CANDIDATES if predicted[dt] < THRESHOLD]
    chosen = max(eligible) if eligible else None
    measured_at_chosen = {
        run["dt"]: run["relative_error"] for run in report["runs"]
    }

    print(f"Richardson (order {ORDER}) estimate at dt = {RICHARDSON_FINE}: "
          f"{richardson:.6e}")
    print(f"{'candidate dt':>14} {'predicted error':>18} {'measured error':>18}")
    for dt in CANDIDATES:
        measured = measured_at_chosen.get(dt)
        measured_text = f"{measured:.6e}" if measured is not None else "-"
        print(f"{dt:>14} {predicted[dt]:>18.6e} {measured_text:>18}")
    print(f"threshold = {THRESHOLD:.1e}")
    print(f"chosen dt = {chosen}")
    print(f"  predicted error = {predicted[chosen]:.6e}")
    print(f"  measured  error = {measured_at_chosen[chosen]:.6e}")

    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    ax.loglog(measured_dt, measured_error, "o", color="#1f77b4", markersize=9,
              label="measured")

    fit_x = np.linspace(measured_dt.min() * 0.95, measured_dt.max() * 1.05, 60)
    ax.loglog(fit_x, np.exp(intercept) * fit_x**slope, "-", color="#1f77b4",
              linewidth=1.2, label=f"fit, slope {slope:.2f}")

    predicted_dt = np.array(CANDIDATES)
    predicted_err = np.array([predicted[dt] for dt in predicted_dt])
    ax.loglog(predicted_dt, predicted_err, "s", markerfacecolor="none",
              markeredgecolor="#ff7f0e", markersize=9,
              label=f"predicted, slope {ORDER}")

    ax.axhline(THRESHOLD, color="0.4", linestyle="--", linewidth=1.1,
               label=f"threshold {THRESHOLD:.0e}")

    ax.plot([chosen], [predicted[chosen]], "*", color="#d62728", markersize=18,
            zorder=5, label=f"chosen dt = {chosen}")
    ax.annotate(
        f"chosen $\\Delta t$ = {chosen}\n"
        f"predicted {predicted[chosen]:.2e}\n"
        f"measured  {measured_at_chosen[chosen]:.2e}",
        xy=(chosen, predicted[chosen]), xytext=(24, -34),
        textcoords="offset points", fontsize=8.5, color="#d62728",
        bbox=dict(facecolor="white", edgecolor="#d62728", alpha=0.9, pad=3.0),
    )

    for dt, error in zip(measured_dt, measured_error):
        ax.annotate(f"{error:.2e}", xy=(dt, error), xytext=(-6, 9),
                    textcoords="offset points", fontsize=8, ha="right")

    ax.set_xlabel("time step $\\Delta t$")
    ax.set_ylabel("relative $\\omega$ error at $t = 2$")
    ax.set_title("Random-flow convergence, RK4, fourth-order Richardson choice")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
