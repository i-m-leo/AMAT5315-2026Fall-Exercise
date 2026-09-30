#!/usr/bin/env python3
"""Time forward- vs reverse-mode gradients of an LJ cluster energy.

For each cluster size the energy is

    U = sum_{i<j} 4 (r_ij^-12 - r_ij^-6)

with r_ij the distance between atoms i and j.  Reverse mode computes the
full 3N-component gradient (the forces) in one backward pass.  Forward
mode propagates one tangent at a time, so a full gradient needs 3N passes;
we time a few directions and scale.

Reference: analytic forces f_i = 24 (2 r_ij^-14 - r_ij^-8) (r_i - r_j),
i.e. the gradient is sum over j of the pairwise force.

Outputs artifacts/ad/scaling.png and prints, per size, the largest
relative error of each mode against the analytic forces and both
time ratios.
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "ad" / "scaling.png"

SIZES = [64, 128, 256, 512, 1024]
SEED = 2026
TIMED_DIRECTIONS = 8  # forward-mode passes actually run, then scaled


def cluster(n, seed):
    """Cubic-lattice cluster with Gaussian displacements (sigma = 0.05)."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n ** (1 / 3)))
    grid = np.arange(side) * 2.0 ** (1.0 / 6.0)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)[:n]
    return pos + rng.normal(0.0, 0.05, pos.shape)


def energy_flat(x):
    """U from the flat 3N coordinate vector, JIT-compiled."""
    pos = x.reshape(-1, 3)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d * d).sum(-1)
    iu = jnp.triu_indices(pos.shape[0], k=1)
    r6 = r2[iu] ** -3
    return 4.0 * jnp.sum(r6 * r6 - r6)


def analytic_forces(pos):
    """Full 3N gradient of U, closed form; grad U = -forces."""
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d * d).sum(-1)
    np.fill_diagonal(r2, np.inf)
    r2 = r2[..., None]
    pair = 24.0 * (2.0 * r2**-7 - r2**-4) * d  # force on i from j
    return pair.sum(axis=1).ravel()


def rel_error(grad, pos):
    """Largest abs error over the largest force component."""
    ref = analytic_forces(pos)
    return float(np.abs(grad + ref).max() / np.abs(ref).max())


# ----------------------------------------------------------------------
results = []
for n in SIZES:
    pos = cluster(n, SEED)
    x = jnp.asarray(pos.ravel())
    en = jax.jit(energy_flat)
    en(x)  # compile
    grad_rev = jax.grad(energy_flat)
    grad_fwd = jax.jvp(energy_flat, (x,), (jnp.ones_like(x),))

    # analytic forces in float64
    ref = analytic_forces(pos)

    # --- reverse mode: one pass gives all 3N components ---------------
    t0 = time.perf_counter()
    g_rev = grad_rev(x).block_until_ready()
    t_rev = time.perf_counter() - t0

    # --- forward mode: 3N passes for the full gradient ----------------
    # time TIMED_DIRECTIONS unit directions, then scale by 3N / timed
    dirs = np.eye(3 * n)[:TIMED_DIRECTIONS]
    t_start = time.perf_counter()
    for k in range(TIMED_DIRECTIONS):
        _, g_k = jax.jvp(energy_flat, (x,), (jnp.asarray(dirs[k]),))
    g_k.block_until_ready()
    t_fwd_timed = (time.perf_counter() - t_start) / TIMED_DIRECTIONS
    t_fwd = t_fwd_timed * (3 * n)

    # full forward gradient for the error check: VJP-free, one call per
    # column basis direction, stacked to the full 3N Jacobian
    _, lin = jax.linearize(energy_flat, x)
    g_fwd_all = np.stack([
        np.asarray(lin(jnp.eye(3 * n)[k])) for k in range(3 * n)
    ])

    err_rev = rel_error(np.asarray(g_rev), pos)
    err_fwd = rel_error(g_fwd_all, pos)

    results.append({
        "n": n, "inputs": 3 * n,
        "t_energy": t_fwd_timed, "t_fwd": t_fwd, "t_rev": t_rev,
        "ratio_fwd": t_fwd / t_fwd_timed, "ratio_rev": t_rev / t_fwd_timed,
        "err_fwd": err_fwd, "err_rev": err_rev,
    })
    print(f"N = {n:5d}  inputs = {3*n:5d}  "
          f"t_energy = {t_fwd_timed*1e3:8.3f} ms  "
          f"t_fwd(full 3N) = {t_fwd*1e3:9.2f} ms  "
          f"t_rev = {t_rev*1e3:8.3f} ms  "
          f"err_fwd = {err_fwd:.2e}  err_rev = {err_rev:.2e}")

# ----------------------------------------------------------------------
# plot: gradient time / energy time vs number of inputs, log-log
# ----------------------------------------------------------------------
inputs = [r["inputs"] for r in results]
ratio_fwd = [r["ratio_fwd"] for r in results]
ratio_rev = [r["ratio_rev"] for r in results]

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(inputs, ratio_fwd, "o-", label="forward mode (full gradient)")
ax.loglog(inputs, ratio_rev, "s-", label="reverse mode (one pass)")
ref = np.array(inputs, dtype=float)
ax.loglog(inputs, ref, "k:", lw=1, label=r"slope 1 ($\times$ #inputs)")
ax.axhline(1.0, color="0.6", lw=1, ls="--", label="slope 0 (constant)")

ax.set_xlabel("number of inputs (3N)")
ax.set_ylabel("gradient time / energy time")
ax.set_title("LJ cluster: cost of a full gradient, forward vs reverse mode")
ax.set_xticks(inputs, [str(i) for i in inputs])
ax.legend()
ax.grid(alpha=0.3, which="both")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
