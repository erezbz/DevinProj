"""Force fields. Each returns (forces[N, dim], potential energy) in SI.

Conservative forces define `potential` consistently with `forces = -grad U`;
tests verify this numerically for every force.
"""
from __future__ import annotations

import numpy as np

from .units import G as G_CONST
from .units import k_e as KE_CONST
from .units import si


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

    def __init__(self, k=KE_CONST, softening=0.0, exclude=()):
        self.k = si(k, "J*m/C^2")
        self.eps2 = si(softening, "m") ** 2
        self.exclude = [tuple(p) for p in exclude]  # id pairs with no interaction (same-molecule atoms)

    def compute(self, s):
        q = s.charges
        if s.n < 2 or not np.any(q):
            return np.zeros_like(s.pos), 0.0
        d, r2 = _pairs(s)
        r2 = r2 + self.eps2
        inv_r = 1 / np.sqrt(r2)
        qq = self.k * q[:, None] * q[None, :]
        if self.exclude:
            where = {pid: n for n, pid in enumerate(s.ids)}
            ex = np.array([(where[a], where[b]) for a, b in self.exclude if a in where and b in where], dtype=int)
            if len(ex):
                qq[ex[:, 0], ex[:, 1]] = 0.0
                qq[ex[:, 1], ex[:, 0]] = 0.0
        f = -np.einsum("ij,ijk->ik", qq * inv_r ** 3, d)
        U = 0.5 * float(np.sum(qq * inv_r))
        return f, U


class VerletList:
    """Cell-list pairs within rc + skin, reused until some particle has moved more than skin/2 (exact: no pair
    within rc can be missed). Rebuilt whenever particle IDs, box or rc change, so reactions are safe."""

    def __init__(self):
        self._state = None
        self.rebuilds = 0

    def pairs(self, s, rc, skin):
        st = self._state
        if st is not None and st[3].shape == s.ids.shape and np.array_equal(st[3], s.ids) \
                and np.array_equal(st[4], s.box) and st[5] == rc and skin > 0:
            disp = s.pos - st[2]
            if s.periodic:
                disp -= s.box * np.round(disp / s.box)
            if float(np.max(np.einsum("ij,ij->i", disp, disp))) < (0.5 * skin) ** 2:
                return st[0], st[1]
        i, j = cell_list_pairs(s, rc + skin)
        self._state = (i, j, s.pos.copy(), s.ids.copy(), None if s.box is None else s.box.copy(), rc)
        self.rebuilds += 1
        return i, j


class LennardJones(Force):
    """12-6 Lennard-Jones with per-species epsilon/sigma (Lorentz-Berthelot mixing).

    params: {species_name: (epsilon, sigma)}. Species not listed don't interact.
    The potential is truncated at cutoff*sigma_ij and shifted to zero there.
    """

    def __init__(self, params: dict, cutoff: float = 2.5, pairlist_threshold: int = 400, skin: float = 0.3):
        self.params = {k: (si(e, "J"), si(sg, "m")) for k, (e, sg) in params.items()}
        self.cutoff = cutoff
        self.pairlist_threshold = pairlist_threshold
        self.skin = skin  # Verlet-list skin (units of sigma): exact, the list is rebuilt before any pair can be missed
        self._vl = VerletList()

    @property
    def rebuilds(self):
        return self._vl.rebuilds

    def _compute_pairlist(self, s):
        """O(N) neighbour search with a cell list: only pairs within the cutoff are visited."""
        eps_t, sig_t = self._tables(s)
        smax = float(np.max(np.where(eps_t > 0, sig_t, 0)))
        i, j = self._vl.pairs(s, self.cutoff * smax, self.skin * smax)
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


class HarmonicAngle(Force):
    """Bond-angle springs on id triplets (i, j, k), vertex j: U = k/2 (theta - theta0)^2."""

    def __init__(self, angles, k, theta0):
        self.angles = [tuple(a) for a in angles]
        self.k = si(k, "J")
        self.theta0 = float(theta0)

    def compute(self, s):
        f = np.zeros_like(s.pos)
        where = {pid: n for n, pid in enumerate(s.ids)}
        trip = [(where[a], where[b], where[c]) for a, b, c in self.angles if a in where and b in where and c in where]
        if not trip:
            return f, 0.0
        i, j, k = np.array(trip).T
        u = s.pos[i] - s.pos[j]
        v = s.pos[k] - s.pos[j]
        if s.periodic:
            u -= s.box * np.round(u / s.box)
            v -= s.box * np.round(v / s.box)
        lu = np.linalg.norm(u, axis=1)
        lv = np.linalg.norm(v, axis=1)
        cos = np.clip(np.einsum("ij,ij->i", u, v) / (lu * lv), -1.0, 1.0)
        theta = np.arccos(cos)
        sin = np.maximum(np.sqrt(1 - cos ** 2), 1e-12)
        pref = (self.k * (theta - self.theta0) / sin)[:, None]
        fi = pref * (v / (lu * lv)[:, None] - cos[:, None] * u / (lu ** 2)[:, None])
        fk = pref * (u / (lu * lv)[:, None] - cos[:, None] * v / (lv ** 2)[:, None])
        np.add.at(f, i, fi)
        np.add.at(f, k, fk)
        np.add.at(f, j, -(fi + fk))
        return f, 0.5 * self.k * float(np.sum((theta - self.theta0) ** 2))


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


def _erfc(x):
    try:
        from scipy.special import erfc
        return erfc(x)
    except ImportError:  # pragma: no cover
        import math
        return np.vectorize(math.erfc)(x)


class Ewald(Force):
    """Exact periodic electrostatics by Ewald summation (3D orthorhombic box, neutral system).

    U = real-space erfc sum (minimum image, r < rcut) + reciprocal sum over k-vectors + self term.
    Defaults give ~1e-8 relative accuracy: rcut = L_min/2, alpha = 4/rcut, |n| <= kmax = 12.
    """

    def __init__(self, k=KE_CONST, alpha=None, rcut=None, kmax=12):
        self.k = si(k, "J*m/C^2")
        self.alpha = None if alpha is None else si(alpha, "1/m")
        self.rcut = None if rcut is None else si(rcut, "m")
        self.kmax = kmax
        self._kcache = None

    def _kvectors(self, box, alpha):
        key = (tuple(box), alpha, self.kmax)
        if self._kcache is None or self._kcache[0] != key:
            n = np.arange(-self.kmax, self.kmax + 1)
            N = np.stack(np.meshgrid(n, n, n, indexing="ij"), -1).reshape(-1, 3)
            N = N[np.sum(N ** 2, axis=1) <= self.kmax ** 2]
            N = N[np.any(N != 0, axis=1)]
            kv = 2 * np.pi * N / box
            k2 = np.sum(kv ** 2, axis=1)
            self._kcache = (key, kv, np.exp(-k2 / (4 * alpha ** 2)) / k2)
        return self._kcache[1], self._kcache[2]

    def compute(self, s):
        if s.dim != 3 or not s.periodic:
            raise ValueError("Ewald needs a 3D periodic box")
        q = s.charges
        f = np.zeros_like(s.pos)
        if not np.any(q):
            return f, 0.0
        if abs(q.sum()) > 1e-9 * np.sum(np.abs(q)):
            raise ValueError("Ewald summation here requires a charge-neutral system")
        box = s.box
        rcut = self.rcut or 0.5 * float(np.min(box))
        a = self.alpha or 4.0 / rcut
        d, r2 = _pairs(s)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        qq = q[:, None] * q[None, :]
        inside = r < rcut
        erfc_r = np.where(inside, _erfc(a * np.where(inside, r, 0)), 0.0)
        U_real = 0.5 * float(np.sum(np.where(inside, qq * erfc_r / r, 0)))
        coef = np.where(inside, qq * (erfc_r / r + 2 * a / np.sqrt(np.pi) * np.exp(-(a * r) ** 2)) / r2, 0)
        f -= np.einsum("ij,ijk->ik", coef, d)
        kv, w = self._kvectors(box, a)
        V = float(np.prod(box))
        phase = s.pos @ kv.T                     # (N, K)
        S = (q[:, None] * np.exp(1j * phase)).sum(0)
        U_rec = 2 * np.pi / V * float(np.sum(w * np.abs(S) ** 2))
        im = np.imag(np.conj(S)[None, :] * np.exp(1j * phase))  # (N, K)
        f += 4 * np.pi / V * q[:, None] * ((w[None, :] * im) @ kv)
        U_self = -a / np.sqrt(np.pi) * float(np.sum(q ** 2))
        return self.k * f, self.k * (U_real + U_rec + U_self)


class DampedShiftedCoulomb(Force):
    """Cutoff electrostatics (damped shifted force, Fennell & Gezelter 2006); O(N) via cell lists for large N.

    Pair potential erfc(a r)/r shifted so both energy and force vanish at rcut, plus the
    matching self term. Approximates Ewald for neutral condensed systems at a fraction of the
    cost; compare against Ewald on a snapshot to quantify the error for your system.
    """

    def __init__(self, rcut, alpha=None, k=KE_CONST, pairlist_threshold=400, exclude=(), skin=None):
        self.pairlist_threshold = pairlist_threshold
        self.exclude = [tuple(p) for p in exclude]  # id pairs with no interaction (same-molecule atoms)
        self.k = si(k, "J*m/C^2")
        self.rcut = si(rcut, "m")
        self.alpha = 2.0 / self.rcut if alpha is None else si(alpha, "1/m")
        self.skin = 0.15 * self.rcut if skin is None else si(skin, "m")  # Verlet-list skin (exact)
        self._vl = VerletList()

    def compute(self, s):
        q = s.charges
        f = np.zeros_like(s.pos)
        if not np.any(q):
            return f, 0.0
        a, rc = self.alpha, self.rcut
        c = 2 * a / np.sqrt(np.pi)
        erc = float(_erfc(np.array(a * rc)))
        fshift = erc / rc ** 2 + c * np.exp(-(a * rc) ** 2) / rc
        where = {pid: n for n, pid in enumerate(s.ids)} if self.exclude else {}
        ex = np.array([(where[a], where[b]) for a, b in self.exclude if a in where and b in where], dtype=int).reshape(-1, 2)
        if s.n >= self.pairlist_threshold:
            i, j = self._vl.pairs(s, rc, self.skin)
            if len(ex):
                lo, hi = np.minimum(ex[:, 0], ex[:, 1]), np.maximum(ex[:, 0], ex[:, 1])
                keep = ~np.isin(i * s.n + j, lo * s.n + hi)
                i, j = i[keep], j[keep]
            d = s.pos[j] - s.pos[i]
            if s.periodic:
                d -= s.box * np.round(d / s.box)
            r = np.sqrt(np.einsum("ij,ij->i", d, d))
            inside = r < rc
            i, j, d, r = i[inside], j[inside], d[inside], r[inside]
            er = _erfc(a * r)
            qq = q[i] * q[j]
            U = float(np.sum(qq * (er / r - erc / rc + fshift * (r - rc))))
            U -= float(np.sum(q ** 2)) * (erc / (2 * rc) + a / np.sqrt(np.pi))
            fv = (qq * ((er / r ** 2 + c * np.exp(-(a * r) ** 2) / r) - fshift) / r)[:, None] * d
            np.add.at(f, i, -fv)
            np.add.at(f, j, fv)
            return self.k * f, self.k * U
        d, r2 = _pairs(s)
        r = np.sqrt(r2)
        inside = r < rc
        rr = np.where(inside, r, 1.0)
        er = _erfc(a * rr)
        qq = np.where(inside, q[:, None] * q[None, :], 0.0)
        if len(ex):
            qq[ex[:, 0], ex[:, 1]] = 0.0
            qq[ex[:, 1], ex[:, 0]] = 0.0
        U = 0.5 * float(np.sum(qq * (er / rr - erc / rc + fshift * (rr - rc))))
        U -= float(np.sum(q ** 2)) * (erc / (2 * rc) + a / np.sqrt(np.pi))
        fmag = qq * ((er / rr ** 2 + c * np.exp(-(a * rr) ** 2) / rr) - fshift) / rr
        f = -self.k * np.einsum("ij,ijk->ik", fmag, d)
        return f, self.k * U
