#!/usr/bin/env python3
"""Compare hand-written forward/reverse AD with centered finite differences.

Same straight-line program as ad_hand.py (a = r^-6, b = a^2, c = b - a,
U = 4c), now evaluated over a grid of separations.  The reference is the
analytic derivative dU/dr = 24 (r^-7 - 2 r^-13).

Outputs artifacts/ad/modes.png and prints the maximum absolute error of
each method.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "ad" / "modes.png"

R_MIN, R_MAX, N = 0.95, 2.5, 601
H = 1e-6

r = np.linspace(R_MIN, R_MAX, N)

# ----------------------------------------------------------------------
# the program, evaluated on the grid (primal values)
# ----------------------------------------------------------------------
a = r**-6.0
b = a**2
c = b - a
u = 4.0 * c

# ----------------------------------------------------------------------
# forward mode: seed tr = 1 at r, propagate node by node
# ----------------------------------------------------------------------
ta = -6.0 * r**-7 * 1.0
tb = 2.0 * a * ta
tc = tb - ta
dUdr_forward = 4.0 * tc

# ----------------------------------------------------------------------
# reverse mode: seed aU = 1 at U, walk backwards, fan-in at a
# ----------------------------------------------------------------------
ac = 4.0 * 1.0
ab = ac * 1.0
aa = ac * (-1.0) + ab * 2.0 * a
dUdr_reverse = -6.0 * r**-7 * aa

# ----------------------------------------------------------------------
# analytic reference and centered finite differences
# ----------------------------------------------------------------------
def u_of(x):
    """U(x) through the same node sequence (used by finite differences)."""
    ax = x**-6.0
    return 4.0 * (ax**2 - ax)


dUdr_analytic = 24.0 * (r**-7.0 - 2.0 * r**-13.0)
dUdr_fd = (u_of(r + H) - u_of(r - H)) / (2.0 * H)

err_forward = np.abs(dUdr_forward - dUdr_analytic)
err_reverse = np.abs(dUdr_reverse - dUdr_analytic)
err_fd = np.abs(dUdr_fd - dUdr_analytic)

# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
print(f"n = {N} points, r in [{R_MIN}, {R_MAX}], h = {H}")
print(f"max |forward - analytic|  = {err_forward.max():.3e}")
print(f"max |reverse - analytic|  = {err_reverse.max():.3e}")
print(f"max |fd       - analytic| = {err_fd.max():.3e}")

# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

ax1.plot(r, dUdr_analytic, "k-", lw=3, alpha=0.35, label="analytic (reference)")
ax1.plot(r, dUdr_forward, "C0--", lw=1.5, label="forward mode (hand)")
ax1.plot(r, dUdr_reverse, "C1.", ms=3, label="reverse mode (hand)")
ax1.plot(r, dUdr_fd, "C2x", ms=3, label=f"centered FD (h = {H:g})")
ax1.set_ylabel(r"$dU/dr$")
ax1.set_title(r"Lennard-Jones $U(r) = 4(r^{-12} - r^{-6})$: derivative by mode")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.loglog(r, err_forward, "C0-", label="forward mode")
ax2.loglog(r, err_reverse, "C1--", label="reverse mode")
ax2.loglog(r, err_fd, "C2-", label="centered FD")
ax2.set_xlabel(r"separation $r$")
ax2.set_ylabel("absolute error vs analytic")
ax2.set_title("error of each mode against the analytic derivative")
ax2.legend()
ax2.grid(alpha=0.3, which="both")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
