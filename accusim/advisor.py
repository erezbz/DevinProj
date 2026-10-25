"""Pre-flight time-step advice from the physical timescales present in a system."""
from __future__ import annotations

import numpy as np

from .forces import Coulomb, Gravity, HarmonicBond, LennardJones


def timescales(sim) -> dict:
    """Shortest characteristic times (s) for each kind of physics present."""
    s = sim.system
    out = {}
    if s.n >= 2:
        d = s.displacements()
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        np.fill_diagonal(r, np.inf)
        m = s.masses
        for f in sim.forces:
            if isinstance(f, Gravity):
                M = m[:, None] + m[None, :]
                out["gravity (orbital)"] = float(np.min(np.sqrt((r ** 2 + f.eps2) ** 1.5 / (f.G * M))))
            elif isinstance(f, Coulomb) and np.any(s.charges):
                q = s.charges
                k = np.abs(f.k * q[:, None] * q[None, :])
                mu = m[:, None] * m[None, :] / (m[:, None] + m[None, :])
                with np.errstate(divide="ignore"):
                    t = np.sqrt(mu * (r ** 2 + f.eps2) ** 1.5 / np.where(k > 0, k, np.nan))
                out["electrostatic"] = float(np.nanmin(t))
            elif isinstance(f, LennardJones):
                eps_t, sig_t = f._tables(s)
                for i, sp in enumerate(s.species):
                    if eps_t[i, i] > 0:
                        t = sig_t[i, i] * np.sqrt(sp.mass / eps_t[i, i])
                        out["Lennard-Jones"] = min(out.get("Lennard-Jones", np.inf), float(t))
            elif isinstance(f, HarmonicBond) and f.bonds:
                out["bond vibration"] = float(2 * np.pi * np.sqrt(np.min(m) / 2 / f.k))
        speed = np.max(np.linalg.norm(s.vel, axis=1))
        if speed > 0:
            out["closest approach / speed"] = float(np.min(r) / speed)
    return out


def suggest_dt(sim, steps_per_timescale: int = 100) -> float:
    ts = timescales(sim)
    if not ts:
        raise ValueError("no timescale found; choose dt manually")
    return min(ts.values()) / steps_per_timescale
