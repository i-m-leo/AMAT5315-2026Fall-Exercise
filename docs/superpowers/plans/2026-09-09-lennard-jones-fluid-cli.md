# Lennard–Jones fluid CLI implementation plan

1. Preserve the existing dimer API and shared `Integrator` trait.
2. Add periodic fluid state, shifted Lennard–Jones forces, deterministic
   initialization, and velocity-Verlet production stepping.
3. Add naive/cell-list force-method equality tests for cutoff, boundaries,
   perturbed lattices, and two-cell-wide boxes before changing the force path.
4. Add artifact serialization, recomputing checks, RDF rendering, and the
   `run`, `check`, and `video` CLI commands.
5. Add release tests, benchmark/scaling evidence, heating artifacts, and the
   GitHub Pages workflow.
