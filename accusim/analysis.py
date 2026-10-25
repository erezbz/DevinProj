"""Structural and transport observables computed from a System or a Trajectory."""
from __future__ import annotations

import numpy as np


def rdf(system, rmax, bins=100, species=None):
    """Radial distribution function g(r) in a periodic box (minimum image). Returns (r, g)."""
    s = system
    idx = np.arange(s.n) if species is None else np.flatnonzero(s.sp == s.species_index(species))
    if not s.periodic:
        raise ValueError("rdf needs a periodic box (for normalisation)")
    if rmax > 0.5 * np.min(np.broadcast_to(s.box, (s.dim,))):
        raise ValueError("rmax must be at most half the box")
    p = s.pos[idx]
    d = p[None, :, :] - p[:, None, :]
    d -= s.box * np.round(d / s.box)
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))[np.triu_indices(len(idx), 1)]
    hist, edges = np.histogram(r, bins=bins, range=(0, rmax))
    V = float(np.prod(np.broadcast_to(s.box, (s.dim,))))
    n = len(idx)
    shell = (4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)) if s.dim == 3 else (np.pi * (edges[1:] ** 2 - edges[:-1] ** 2))
    ideal = shell * n * (n - 1) / 2 / V
    return 0.5 * (edges[1:] + edges[:-1]), hist / ideal


def msd(trajectory):
    """Mean squared displacement vs time from recorded positions (fixed particle set, unwrapped coords)."""
    p0 = trajectory.positions[0]
    t = np.asarray(trajectory.t)
    return t - t[0], np.array([np.mean(np.sum((p - p0) ** 2, axis=1)) for p in trajectory.positions])
