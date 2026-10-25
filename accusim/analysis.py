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


def msd(trajectory, box=None):
    """Mean squared displacement vs time from recorded positions (fixed particle set).

    With a periodic `box`, frames are unwrapped by minimum-image steps, so frames must be
    recorded often enough that no particle moves half a box between them.
    """
    t = np.asarray(trajectory.t)
    frames = trajectory.positions
    disp = np.zeros_like(frames[0])
    out = [0.0]
    for a, b in zip(frames[:-1], frames[1:]):
        step = b - a
        if box is not None:
            step -= box * np.round(step / box)
        disp = disp + step
        out.append(float(np.mean(np.sum(disp ** 2, axis=1))))
    return t - t[0], np.array(out)


def pressure(sim, h=1e-6):
    """Instantaneous pressure P = 2K/(dV) - dU/dV for any set of forces.

    dU/dV is taken by uniformly rescaling the box and all positions (central
    difference), so it works for every force type without per-force virials.
    """
    s = sim.system
    if s.box is None:
        raise ValueError("pressure needs a box")
    box0, pos0 = np.array(s.box, dtype=float), s.pos.copy()
    V0 = float(np.prod(np.broadcast_to(box0, (s.dim,))))
    U = []
    for sign in (+1, -1):
        lam = (1 + sign * h) ** (1 / s.dim)
        s.box, s.pos = box0 * lam, pos0 * lam
        U.append(sum(f.compute(s)[1] for f in sim.forces if not f.external))
    s.box, s.pos = (box0 if np.ndim(box0) else float(box0)), pos0
    dUdV = (U[0] - U[1]) / (2 * h * V0)
    return 2 * s.kinetic_energy() / (s.dim * V0) - dUdV


def wall_pressure(system, duration):
    """Mechanical pressure on reflecting walls: momentum delivered (System.wall_impulse,
    reset it to 0 before measuring) per unit wall area per unit time."""
    if system.box is None or system.periodic:
        raise ValueError("wall_pressure needs a reflecting (non-periodic) box")
    V = float(np.prod(system.box))
    area = 2 * V * float(np.sum(1 / system.box))
    return system.wall_impulse / (area * duration)


def vacf(trajectory, max_lag=None):
    """Velocity autocorrelation <v(0).v(t)> averaged over particles and time origins (fixed particle set).
    Returns (lag times, C(t))."""
    V = np.asarray(trajectory.velocities)
    if V.ndim != 3:
        raise ValueError("vacf needs recorded velocities of a constant particle set")
    n = len(V)
    max_lag = n // 2 if max_lag is None else max_lag
    C = np.array([np.mean(np.einsum("tij,tij->ti", V[:n - k], V[k:])) for k in range(max_lag)])
    t = np.asarray(trajectory.t)
    return t[:max_lag] - t[0], C


def green_kubo_diffusion(trajectory, max_lag=None):
    """Self-diffusion D = (1/d) * integral of the VACF (trapezoid), an independent route to the MSD slope."""
    t, C = vacf(trajectory, max_lag)
    dim = np.asarray(trajectory.velocities).shape[2]
    return float(np.sum(0.5 * (C[1:] + C[:-1]) * np.diff(t))) / dim
