"""Energy minimisation with FIRE (Bitzek et al., PRL 97, 170201, 2006).

Used to relax hand-built starting structures (no overlapping atoms, bonds at
rest length) before dynamics. It moves positions only; velocities are zeroed.
"""
from __future__ import annotations

import numpy as np


def minimize(sim, ftol=None, max_steps=20000, dt0=None):
    """Relax sim.system to a local energy minimum. Returns (U_final, max_force, steps)."""
    s = sim.system
    dt = dt0 if dt0 is not None else sim.dt
    dt_max = 10 * dt
    alpha, n_pos = 0.1, 0
    m = s.masses[:, None]
    v = np.zeros_like(s.pos)
    F, U = sim.potential()
    if ftol is None:
        ftol = 1e-9 * max(float(np.max(np.abs(F))), 1e-300)
    for step in range(max_steps):
        fmax = float(np.max(np.linalg.norm(F, axis=1))) if s.n else 0.0
        if fmax < ftol:
            break
        P = float(np.sum(F * v))
        if P > 0:
            fn = np.linalg.norm(F)
            v = (1 - alpha) * v + alpha * np.linalg.norm(v) * F / (fn if fn > 0 else 1.0)
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            dt *= 0.5
            alpha, n_pos = 0.1, 0
        v = v + dt * F / m
        s.pos += dt * v
        s.apply_boundaries()
        F, U = sim.potential()
    s.vel[:] = 0.0
    sim.update_forces()
    return U, float(np.max(np.linalg.norm(F, axis=1))) if s.n else 0.0, step
