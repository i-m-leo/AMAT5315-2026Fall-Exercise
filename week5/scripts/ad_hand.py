#!/usr/bin/env python3
"""Forward- and reverse-mode AD by hand for the Lennard-Jones pair energy.

U(r) = 4 (r^-12 - r^-6) evaluated through the straight-line program

    a = r^-6,   b = a^2,   c = b - a,   U = 4c

at r = 1.3 in 64-bit floats.  Tangents come from the forward pass,
adjoints from the reverse pass; both are checked against jax.grad.
"""
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "ad" / "derivatives.json"

R = 1.3


def energy(r):
    """The straight-line program, written once; AD modes re-walk it by hand."""
    a = r**-6
    b = a**2
    c = b - a
    return 4.0 * c


# ----------------------------------------------------------------------
# forward mode: tangent of each node, one node at a time
# ----------------------------------------------------------------------
# node   primal            tangent
# r      1.3               1.0   (seed: d/dr)
# a      r^-6              -6 r^-7
# b      a^2               2 a * ta
# c      b - a             tb - ta
# U      4 c               4 tc

r = R
a = r**-6
b = a**2
c = b - a
u = 4.0 * c

tr = 1.0
ta = -6.0 * r**-7          # d(r^-6)/dr = -6 r^-7
tb = 2.0 * a * ta          # product rule on a^2
tc = tb - ta               # subtraction: tangents subtract
tu = 4.0 * tc              # constant factor passes through

tangents = {"r": tr, "a": ta, "b": tb, "c": tc, "U": tu}

# ----------------------------------------------------------------------
# reverse mode: adjoint of each node, walking the program backwards
# ----------------------------------------------------------------------
# U = 4c          dU/dU = 1            -> aU = 1
# c = b - a       dU/dc = 4            -> ac = 4
#                 dU/db = ac * 1       -> ab = 4
#                 dU/da = ac * (-1)    -> a_a_partial = -4
# b = a^2         dU/db = ab           -> abar = 4
#                 dU/da += ab * 2a     -> a_a_partial += 8a
# a = r^-6        dU/da = -4 + 8a      -> aa = -4 + 8a
#                 dU/dr = aa * (-6 r^-7)
# r               dr = aa * (-6 r^-7)

au = 1.0
ac = 4.0 * au
ab = ac * 1.0
aa = ac * (-1.0)           # from c = b - a
ab *= 1.0                  # adjoint of b consumed by c
aa += ab * 2.0 * a         # from b = a^2
ar = aa * (-6.0 * r**-7)   # from a = r^-6

adjoints = {"r": ar, "a": aa, "b": ab, "c": ac, "U": au}

# ----------------------------------------------------------------------
# JAX's own reverse-mode gradient
# ----------------------------------------------------------------------
jax_grad = float(jax.grad(energy)(jnp.float64(R)))

# closed form for a final sanity check: dU/dr = 24 (r^-7 - 2 r^-13)
closed = 24.0 * (r**-7 - 2.0 * r**-13)

result = {
    "r": r,
    "energy": u,
    "tangents": tangents,
    "adjoints": adjoints,
    "jax_grad": jax_grad,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    json.dump(result, f, indent=2)
    f.write("\n")

print(json.dumps(result, indent=2))
print(f"closed form dU/dr = {closed!r}")
assert abs(tu - ar) < 1e-15, "forward and reverse modes disagree"
assert abs(jax_grad - tu) < 1e-15, "JAX disagrees with the hand pass"
assert abs(closed - tu) < 1e-12, "closed form disagrees"
print("forward == reverse == jax == closed form")
