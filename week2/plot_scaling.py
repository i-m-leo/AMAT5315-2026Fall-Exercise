#!/usr/bin/env python3
"""Plot the measured naive and cell-list MD scaling timings."""
import matplotlib.pyplot as plt

N = [100, 400, 1600]
NAIVE = [0.02, 0.20, 3.13]
CELLS = [0.02, 0.09, 0.39]
STEPS = 100 + 500

fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
ax.plot(N, [t / STEPS for t in NAIVE], "o-", linewidth=2, label="naive")
ax.plot(N, [t / STEPS for t in CELLS], "o-", linewidth=2, label="cells")
ax.set_xlabel("N particles")
ax.set_ylabel("seconds per step")
ax.set_title("Lennard–Jones force scaling")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(N, [str(n) for n in N])
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.savefig("scaling.png", dpi=160)
