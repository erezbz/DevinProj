"""Force fields. Each returns (forces[N, dim], potential energy) in SI.

Conservative forces define `potential` consistently with `forces = -grad U`;
tests verify this numerically for every force.
"""
from __future__ import annotations

import numpy as np

from .units import G as G_CONST, k_e as KE_CONST, si


class Force:
    conservative = True
    external = False  # breaks momentum conservation (fixed field, drag)
    central = True    # pairwise central: conserves angular momentum

    def compute(self, system):
        raise NotImplementedError


def _pairs(system):
    d = system.displacements()
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, np.inf)
    return d, r2


class Gravity(Force):
    """Newtonian gravity between all particles, optional Plummer softening."""

    def __init__(self, G=G_CONST, softening=0.0):
        self.G = si(G, "m^3/kg/s^2")
        self.eps2 = si(softening, "m") ** 2

    def compute(self, s):
        if s.n < 2:
            return np.zeros_like(s.pos), 0.0
        d, r2 = _pairs(s)
        r2 = r2 + self.eps2
        m = s.masses
        inv_r = 1 / np.sqrt(r2)
        mm = self.G * m[:, None] * m[None, :]
        f = np.einsum("ij,ijk->ik", mm * inv_r ** 3, d)
        U = -0.5 * float(np.sum(mm * inv_r))
        return f, U


class Coulomb(Force):
    """Electrostatics between charged particles (k = 1/4 pi eps0 by default)."""

    def __init__(self, k=KE_CONST, softening=0.0):
        self.k = si(k, "J*m/C^2")
        self.eps2 = si(softening, "m") ** 2

    def compute(self, s):
        q = s.charges
        if s.n < 2 or not np.any(q):
            return np.zeros_like(s.pos), 0.0
        d, r2 = _pairs(s)
        r2 = r2 + self.eps2
        inv_r = 1 / np.sqrt(r2)
        qq = self.k * q[:, None] * q[None, :]
        f = -np.einsum("ij,ijk->ik", qq * inv_r ** 3, d)
        U = 0.5 * float(np.sum(qq * inv_r))
        return f, U


class LennardJones(Force):
    """12-6 Lennard-Jones with per-species epsilon/sigma (Lorentz-Berthelot mixing).

    params: {species_name: (epsilon, sigma)}. Species not listed don't interact.
    The potential is truncated at cutoff*sigma_ij and shifted to zero there.
    """

    def __init__(self, params: dict, cutoff: float = 2.5, pairlist_threshold: int = 400):
        self.params = {k: (si(e, "J"), si(sg, "m")) for k, (e, sg) in params.items()}
        self.cutoff = cutoff
        self.pairlist_threshold = pairlist_threshold

    def _compute_pairlist(self, s):
        """O(N) neighbour search with a cell list: only pairs within the cutoff are visited."""
        eps_t, sig_t = self._tables(s)
        rc = self.cutoff * float(np.max(np.where(eps_t > 0, sig_t, 0)))
        i, j = cell_list_pairs(s, rc)
        f = np.zeros_like(s.pos)
        if not len(i):
            return f, 0.0
        d = s.pos[j] - s.pos[i]
        if s.periodic:
            d -= s.box * np.round(d / s.box)
        r2 = np.einsum("ij,ij->i", d, d)
        eps = eps_t[s.sp[i], s.sp[j]]
        sig = sig_t[s.sp[i], s.sp[j]]
        mask = (r2 < (self.cutoff * sig) ** 2) & (eps > 0)
        d, r2, eps, sig, i, j = d[mask], r2[mask], eps[mask], sig[mask], i[mask], j[mask]
        sr6 = (sig ** 2 / r2) ** 3
        src6 = (1 / self.cutoff ** 2) ** 3
        U = float(np.sum(4 * eps * (sr6 ** 2 - sr6 - (src6 ** 2 - src6))))
        coef = -24 * eps * (2 * sr6 ** 2 - sr6) / r2
        fij = coef[:, None] * d
        np.add.at(f, i, fij)
        np.add.at(f, j, -fij)
        return f, U

    def _tables(self, s):
        n = len(s.species)
        eps = np.zeros((n, n))
        sig = np.ones((n, n))
        for i, a in enumerate(s.species):
            for j, b in enumerate(s.species):
                if a.name in self.params and b.name in self.params:
                    ea, sa = self.params[a.name]
                    eb, sb = self.params[b.name]
                    eps[i, j] = np.sqrt(ea * eb)
                    sig[i, j] = 0.5 * (sa + sb)
        return eps, sig

    def compute(self, s):
        if s.n < 2:
            return np.zeros_like(s.pos), 0.0
        if s.n >= self.pairlist_threshold:
            return self._compute_pairlist(s)
        eps_t, sig_t = self._tables(s)
        eps = eps_t[s.sp[:, None], s.sp[None, :]]
        sig = sig_t[s.sp[:, None], s.sp[None, :]]
        d, r2 = _pairs(s)
        rc2 = (self.cutoff * sig) ** 2
        mask = (r2 < rc2) & (eps > 0)
        sr2 = np.where(mask, sig ** 2 / np.where(mask, r2, 1.0), 0.0)
        sr6 = sr2 ** 3
        src6 = (1 / self.cutoff ** 2) ** 3
        u = np.where(mask, 4 * eps * (sr6 ** 2 - sr6 - (src6 ** 2 - src6)), 0.0)
        # F_i = -dU/dr * (r_i - r_j)/r ; with d = r_j - r_i
        coef = np.where(mask, -24 * eps * (2 * sr6 ** 2 - sr6) / np.where(mask, r2, 1.0), 0.0)
        f = np.einsum("ij,ijk->ik", coef, d)
        return f, 0.5 * float(np.sum(u))


class HarmonicBond(Force):
    """Springs between particle ids: U = k/2 (r - r0)^2."""

    def __init__(self, bonds, k, r0=0.0):
        self.bonds = [tuple(b) for b in bonds]
        self.k = si(k, "N/m")
        self.r0 = si(r0, "m")

    def compute(self, s):
        f = np.zeros_like(s.pos)
        if not self.bonds:
            return f, 0.0
        where = {pid: i for i, pid in enumerate(s.ids)}
        pairs = [(where[a], where[b]) for a, b in self.bonds if a in where and b in where]
        if not pairs:
            return f, 0.0
        i, j = np.array(pairs).T
        d = s.pos[j] - s.pos[i]
        if s.periodic:
            d -= s.box * np.round(d / s.box)
        r = np.linalg.norm(d, axis=1)
        stretch = r - self.r0
        unit = d / np.where(r > 0, r, 1.0)[:, None]
        fij = (self.k * stretch)[:, None] * unit
        np.add.at(f, i, fij)
        np.add.at(f, j, -fij)
        return f, 0.5 * self.k * float(np.sum(stretch ** 2))


class UniformField(Force):
    """Constant gravitational acceleration g (vector, m/s^2) and/or electric field E (V/m)."""

    external = True
    central = False

    def __init__(self, g=None, E=None):
        self.g = None if g is None else np.asarray(si(g, "m/s^2"), float)
        self.E = None if E is None else np.asarray(si(E, "V/m"), float)

    def compute(self, s):
        f = np.zeros_like(s.pos)
        U = 0.0
        if self.g is not None:
            m = s.masses[:, None]
            f += m * self.g
            U -= float(np.sum(m * self.g * s.pos))
        if self.E is not None:
            q = s.charges[:, None]
            f += q * self.E
            U -= float(np.sum(q * self.E * s.pos))
        return f, U


class Drag(Force):
    """Linear drag F = -gamma m v (dissipative: energy is not conserved)."""

    conservative = False
    external = True
    central = False

    def __init__(self, gamma):
        self.gamma = si(gamma, "1/s")

    def compute(self, s):
        return -self.gamma * s.masses[:, None] * s.vel, 0.0


def cell_list_pairs(s, rc):
    """All pairs (i < j) closer than rc, found by binning particles into cells of size >= rc."""
    pos = s.pos
    if s.box is not None:
        lo, span = np.zeros(s.dim), s.box
    else:
        lo = pos.min(axis=0)
        span = np.maximum(pos.max(axis=0) - lo, rc)
    ncell = np.maximum((span // rc).astype(int), 1)
    if s.periodic:
        ncell = np.where(ncell < 3, 1, ncell)
    cell = np.minimum(((pos - lo) / span * ncell).astype(int), ncell - 1)
    cell = np.maximum(cell, 0)
    flat = np.ravel_multi_index(cell.T, ncell)
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    starts = np.searchsorted(sorted_flat, np.arange(np.prod(ncell)))
    ends = np.searchsorted(sorted_flat, np.arange(np.prod(ncell)), side="right")
    offsets = np.array(np.meshgrid(*[[-1, 0, 1]] * s.dim)).reshape(s.dim, -1).T
    I, J = [], []
    for c in range(int(np.prod(ncell))):
        a = order[starts[c]:ends[c]]
        if not len(a):
            continue
        cidx = np.array(np.unravel_index(c, ncell))
        neigh = cidx + offsets
        if s.periodic:
            neigh %= ncell
        else:
            neigh = neigh[np.all((neigh >= 0) & (neigh < ncell), axis=1)]
        nflat = np.unique(np.ravel_multi_index(neigh.T, ncell))
        b = np.concatenate([order[starts[n]:ends[n]] for n in nflat])
        ii, jj = np.meshgrid(a, b, indexing="ij")
        keep = ii < jj
        I.append(ii[keep]); J.append(jj[keep])
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    i, j = np.concatenate(I), np.concatenate(J)
    d = pos[j] - pos[i]
    if s.periodic:
        d -= s.box * np.round(d / s.box)
    close = np.einsum("ij,ij->i", d, d) < rc ** 2
    return i[close], j[close]
