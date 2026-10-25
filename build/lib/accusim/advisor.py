"""Pre-flight time-step advice from the physical timescales present in a system."""
from __future__ import annotations

import numpy as np

from .forces import Coulomb, Gravity, HarmonicBond, LennardJones


def timescales(sim, max_rows: int = 1500) -> dict:
    """Shortest characteristic times (s) for each kind of physics present."""
    s = sim.system
    out = {}
    if s.n >= 2:
        from .tree import BarnesHut
        # rows: all particles for small systems, a random sample for large ones (keeps memory O(N))
        sub = np.arange(s.n) if s.n <= max_rows else np.random.default_rng(0).choice(s.n, max_rows, replace=False)
        d = s.pos[None, :, :] - s.pos[sub][:, None, :]
        if s.periodic:
            d -= s.box * np.round(d / s.box)
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        r[np.arange(len(sub)), sub] = np.inf
        m_all = s.masses
        m = m_all[sub][:, None] * np.ones((1, s.n))
        m_col = np.ones((len(sub), 1)) * m_all[None, :]
        for f in sim.forces:
            if isinstance(f, (Gravity, BarnesHut)):
                M = m + m_col
                out["gravity (orbital)"] = float(np.min(np.sqrt((r ** 2 + f.eps2) ** 1.5 / (f.G * M))))
            elif isinstance(f, Coulomb) and np.any(s.charges):
                q = s.charges
                k = np.abs(f.k * q[sub][:, None] * q[None, :])
                mu = m * m_col / (m + m_col)
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
                out["bond vibration"] = float(2 * np.pi * np.sqrt(np.min(m_all) / 2 / f.k))
        speed = np.max(np.linalg.norm(s.vel, axis=1))
        if speed > 0 and any(not f.external for f in sim.forces):
            eps2 = max([getattr(f, "eps2", 0.0) for f in sim.forces] or [0.0])
            out["closest approach / speed"] = float(np.sqrt(np.min(r) ** 2 + eps2) / speed)
    return out


def suggest_dt(sim, steps_per_timescale: int = 100) -> float:
    ts = timescales(sim)
    if not ts:
        raise ValueError("no timescale found; choose dt manually")
    return min(ts.values()) / steps_per_timescale


def calibrate_dt(sim, trial_duration, energy_tol=1e-6, max_halvings=12):
    """Halve dt until a trial run conserves energy to `energy_tol`; sets and returns sim.dt.

    Only meaningful for conservative, deterministic physics. Returns (dt, measured_drift).
    """
    from .units import si
    if sim.integrator.stochastic or sim.integrator.thermostat or not all(f.conservative for f in sim.forces):
        raise ValueError("calibrate_dt needs conservative, deterministic physics")
    T = si(trial_duration, "s")
    dt = sim.dt if sim.dt is not None else suggest_dt(sim)
    for _ in range(max_halvings + 1):
        trial = sim.clone(dt=dt)
        n = max(int(round(T / dt)), 1)
        tr = trial.run(steps=n, record_every=max(1, n // 200), positions=False)
        E = tr.total_energy
        scale = max(abs(E[0]), max(abs(x) for x in tr.kinetic), max(abs(x) for x in tr.potential), 1e-300)
        drift = float(np.max(np.abs(E - E[0])) / scale)
        if drift <= energy_tol:
            sim.dt = dt
            return dt, drift
        dt /= 2
    raise RuntimeError(f"energy drift {drift:.2e} still above {energy_tol:.0e} after {max_halvings} halvings")
