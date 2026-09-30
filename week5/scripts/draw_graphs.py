#!/usr/bin/env python3
"""Draw the computational graphs JAX records for U(r) and for its gradient.

The graphs are read straight out of JAX with `jax.make_jaxpr`: one node per
traced operation, labelled with the primitive's name, edges wired exactly as
the jaxpr connects them.  For the gradient graph the ops reachable from the
cotangent seed (the literal 1.0) are shaded as the backward/VJP pass; the rest
is the forward pass.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

BASE = Path(__file__).resolve().parent.parent
OUTDIR = BASE / "artifacts" / "ad"


def energy(r):
    a = r**-6
    b = a**2
    c = b - a
    return 4.0 * c


# ----------------------------------------------------------------------
# jaxpr -> nodes + edges
# ----------------------------------------------------------------------
def collect(jaxpr):
    """One node per variable; op nodes labelled with the primitive name."""
    nodes, edges = {}, []

    def ensure(v, label, kind, level=0):
        key = id(v)
        if key not in nodes:
            nodes[key] = {"label": label, "kind": kind, "level": level}
        return key

    for i, v in enumerate(jaxpr.invars):
        ensure(v, "r" if len(jaxpr.invars) == 1 else f"arg{i}", "input")
    for v in jaxpr.constvars:
        ensure(v, str(v.val), "input")

    for eqn in jaxpr.eqns:
        in_ids = []
        for v in eqn.invars:
            if hasattr(v, "val"):  # literal flowing straight into the eqn
                in_ids.append(ensure(v, str(v.val), "input"))
            else:
                in_ids.append(id(v))
        level = max(nodes[i]["level"] for i in in_ids) + 1
        out = eqn.outvars[0]
        out_id = ensure(out, eqn.primitive.name, "op", level)
        nodes[out_id]["prim"] = eqn.primitive.name
        for src in in_ids:
            edges.append((src, out_id))

    out_ids = [id(v) for v in jaxpr.outvars]
    return nodes, edges, out_ids


def mark_backward(nodes, edges):
    """An op is backward iff it is reachable from the cotangent seed (1.0)."""
    succ = {}
    for s, d in edges:
        succ.setdefault(s, []).append(d)
    seeds = [k for k, nd in nodes.items()
             if nd["kind"] == "input" and nd["label"] == "1.0"]
    back, stack = set(), list(seeds)
    while stack:
        k = stack.pop()
        if k in back:
            continue
        back.add(k)
        stack.extend(succ.get(k, []))
    for k, nd in nodes.items():
        if nd["kind"] == "op":
            nd["pass"] = "backward" if k in back else "forward"
    return bool(seeds)


def layout(nodes):
    by_level = {}
    for key, nd in nodes.items():
        by_level.setdefault(nd["level"], []).append(key)
    pos = {}
    for lv, keys in sorted(by_level.items()):
        keys = sorted(keys, key=lambda k: (nodes[k]["kind"], nodes[k]["label"]))
        n = len(keys)
        for j, k in enumerate(keys):
            pos[k] = (lv, 1.0 - (j + 0.5) / n)
    return pos, max(by_level)


def draw(nodes, edges, out_ids, title, path):
    pos, n_levels = layout(nodes)
    has_backward = mark_backward(nodes, edges)
    fig, ax = plt.subplots(figsize=(2.2 * n_levels + 3.0, 7.0))

    for s, d in edges:
        ax.add_patch(FancyArrowPatch(
            pos[s], pos[d], arrowstyle="-|>", mutation_scale=12,
            color="0.55", lw=1.1, shrinkA=11, shrinkB=11, zorder=1))

    for key, nd in nodes.items():
        x, y = pos[key]
        if nd["kind"] == "op":
            color = ("#f39c50" if nd.get("pass") == "backward" else "#4c92d6")
            face = color
        else:
            face = "#c9c9c9"
        ax.scatter([x], [y], s=620, facecolor=face, edgecolor="k",
                   linewidth=0.8, zorder=3)
        ax.text(x, y + 0.055, nd["label"], ha="center", fontsize=9, zorder=4)

    for i, k in enumerate(out_ids):
        x, y = pos[k]
        label = "U" if not has_backward else "dU/dr"
        ax.annotate(label, (x, y), xytext=(x + 0.35, y),
                    fontsize=11, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="k", lw=0.7))
        ax.scatter([x], [y], s=1000, facecolor="none", edgecolor="k",
                   linewidth=1.6, zorder=4)

    handles = [Line2D([], [], marker="o", ls="", color="#4c92d6",
                      markeredgecolor="k", markersize=11, label="forward pass op"),
               Line2D([], [], marker="o", ls="", color="#c9c9c9",
                      markeredgecolor="k", markersize=11, label="input / constant")]
    if has_backward:
        handles.insert(1, Line2D([], [], marker="o", ls="", color="#f39c50",
                                 markeredgecolor="k", markersize=11,
                                 label="backward (VJP) pass op"))
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.6, n_levels + 0.9)
    ax.set_ylim(-0.08, 1.12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


for name, fn in [("graph", energy), ("grad-graph", jax.grad(energy))]:
    jaxpr = jax.make_jaxpr(fn)(jnp.float64(1.3))
    print(f"--- jaxpr for {name} ---")
    print(jaxpr)
    nodes, edges, out_ids = collect(jaxpr)
    title = (r"JAX computational graph for $U(r)$ at $r = 1.3$"
             if name == "graph"
             else r"JAX computational graph for $\partial U/\partial r$ (jax.grad)")
    draw(nodes, edges, out_ids, title, OUTDIR / f"{name}.png")
