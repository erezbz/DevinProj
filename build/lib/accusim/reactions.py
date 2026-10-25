"""Particle reactions with exact conservation of momentum, charge and mass-energy.

Energy released Q = (sum of reactant masses - sum of product masses) c^2 + `energy`.
Products share the available energy in the center-of-mass frame, so momentum is
conserved exactly; if the available energy is negative the reaction cannot occur
(a physical threshold). With a single product the relative kinetic energy is
stored as internal energy of the product (inelastic merge) and tracked.
"""
from __future__ import annotations

import numpy as np

from .units import c as C_LIGHT, si

C2 = C_LIGHT.value ** 2


class Reaction:
    def __init__(self, reactants, products, rate, radius=0.0, energy=0.0, separation=None, name=None):
        self.reactants = tuple([reactants] if isinstance(reactants, str) else reactants)
        self.products = tuple([products] if isinstance(products, str) else products)
        if len(self.reactants) not in (1, 2):
            raise ValueError("reactions need 1 or 2 reactants")
        self.rate = si(rate, "1/s")
        self.radius = si(radius, "m")
        self.energy = si(energy, "J")
        self.separation = None if separation is None else si(separation, "m")
        self.name = name or f"{' + '.join(self.reactants)} -> {' + '.join(self.products) or '0'}"
        if len(self.reactants) == 2 and self.radius <= 0:
            raise ValueError(f"{self.name}: bimolecular reactions need a reaction radius")

    def validate(self, system):
        sp = {s.name: s for s in system.species}
        for n in self.reactants + self.products:
            if n not in sp:
                raise KeyError(f"{self.name}: unknown species '{n}'")
        q_in = sum(sp[n].charge for n in self.reactants)
        q_out = sum(sp[n].charge for n in self.products)
        if not np.isclose(q_in, q_out, rtol=1e-12, atol=1e-30):
            raise ValueError(f"{self.name}: charge not conserved ({q_in:g} C -> {q_out:g} C)")
        if not self.products:
            raise ValueError(f"{self.name}: reactions must have products (mass-energy must go somewhere)")
        m_in = sum(sp[n].mass for n in self.reactants)
        m_out = sum(sp[n].mass for n in self.products)
        self.rest_q = (m_in - m_out) * C2
        self.q_value = self.rest_q + self.energy
        self._sep = self.separation
        if self._sep is None:
            self._sep = max(sum(sp[n].radius for n in self.products[:2]), 0.0)


class ReactionEngine:
    def __init__(self, reactions, system):
        self.reactions = list(reactions)
        for r in self.reactions:
            r.validate(system)
        self.internal = 0.0     # potential-energy jumps absorbed by reaction events
        self.released = 0.0     # rest-mass/chemical energy converted (sum of Q)
        self.released_rest = 0.0  # part of `released` that came from rest mass
        self.events = {r.name: 0 for r in self.reactions}

    def step(self, sim, dt):
        s = sim.system
        changed = False
        busy = np.zeros(s.n, dtype=bool)
        new = []  # (species list, positions, velocities)
        for r in self.reactions:
            p = 1 - np.exp(-r.rate * dt)
            if len(r.reactants) == 1:
                cand = np.flatnonzero((s.sp == s.species_index(r.reactants[0])) & ~busy)
                hits = cand[sim.rng.random(len(cand)) < p]
                for i in hits:
                    if self._fire(sim, r, [i], new):
                        busy[i] = True
            else:
                ia = np.flatnonzero((s.sp == s.species_index(r.reactants[0])) & ~busy)
                ib = np.flatnonzero((s.sp == s.species_index(r.reactants[1])) & ~busy)
                if not len(ia) or not len(ib):
                    continue
                d = s.displacements(ia, ib)
                r2 = np.einsum("ijk,ijk->ij", d, d)
                close = r2 < r.radius ** 2
                if r.reactants[0] == r.reactants[1]:
                    close = np.triu(close, 1)
                pairs = np.argwhere(close)
                sim.rng.shuffle(pairs)
                for a, b in pairs:
                    i, j = ia[a], ib[b]
                    if busy[i] or busy[j] or sim.rng.random() >= p:
                        continue
                    if self._fire(sim, r, [i, j], new):
                        busy[i] = busy[j] = True
            changed = changed or busy.any()
        if not changed:
            return False
        _, u_before = sim.potential()
        s.remove(busy)
        for names, pos, vel, eint in new:
            for n, x, v, e in zip(names, pos, vel, eint):
                s.add(n, x, v, e)
        s.apply_boundaries()
        sim.update_forces()
        # Potential energy jumps when particles appear/disappear; the reacting
        # species' internal energy absorbs it so the books balance exactly.
        self.internal -= sim.pe - u_before
        return True

    def _fire(self, sim, r, idx, new):
        s = sim.system
        m = s.masses[idx]
        v = s.vel[idx]
        x = s.pos[idx].copy()
        if len(idx) == 2:  # unwrap partner next to first reactant
            d = s.displacements([idx[0]], [idx[1]])[0, 0]
            x[1] = x[0] + d
        P = np.sum(m[:, None] * v, axis=0)
        ke_in = 0.5 * float(np.sum(m[:, None] * v ** 2)) + float(np.sum(s.eint[idx]))
        x_cm = np.sum(m[:, None] * x, axis=0) / m.sum()
        mp = np.array([s.species[s.species_index(n)].mass for n in r.products])
        V = P / mp.sum()
        e_rel = ke_in + r.q_value - 0.5 * mp.sum() * float(V @ V)
        if e_rel < 0:
            return False  # below threshold: not enough energy
        k = len(mp)
        if k == 1:
            pos, vel, eint = x_cm[None, :], V[None, :], [e_rel]
        else:
            w = sim.rng.normal(size=(k, s.dim))
            w -= np.sum(mp[:, None] * w, axis=0) / mp.sum()
            ke_w = 0.5 * float(np.sum(mp[:, None] * w ** 2))
            w *= np.sqrt(e_rel / ke_w) if ke_w > 0 else 0.0
            vel = V + w
            dirs = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-300)
            off = 0.5 * r._sep * dirs
            off -= np.sum(mp[:, None] * off, axis=0) / mp.sum()
            pos = x_cm + off
            eint = [0.0] * k
        self.released += r.q_value
        self.released_rest += r.rest_q
        self.events[r.name] += 1
        new.append((list(r.products), pos, vel, eint))
        return True


def gillespie(counts: dict, reactions, volume, t_end, rng=None, max_events=10_000_000):
    """Exact stochastic simulation (Gillespie SSA) of a well-mixed volume.

    reactions: list of (reactants, products, k) with k in 1/s (unimolecular) or
    m^3/s (bimolecular, per pair). Returns (times, {species: counts array}).
    """
    rng = rng or np.random.default_rng()
    V = si(volume, "m^3")
    names = sorted(set(counts) | {n for r in reactions for n in list(r[0]) + list(r[1])})
    x = {n: int(counts.get(n, 0)) for n in names}
    t, times, hist = 0.0, [0.0], {n: [x[n]] for n in names}
    rxs = [(tuple(a), tuple(b), si(k)) for a, b, k in reactions]
    for _ in range(max_events):
        props = []
        for a, b, k in rxs:
            if len(a) == 1:
                props.append(k * x[a[0]])
            elif a[0] == a[1]:
                props.append(k / V * x[a[0]] * (x[a[0]] - 1) / 2)
            else:
                props.append(k / V * x[a[0]] * x[a[1]])
        total = sum(props)
        if total <= 0:
            break
        t += rng.exponential(1 / total)
        if t > t_end:
            break
        j = np.searchsorted(np.cumsum(props), rng.random() * total, side="right")
        for n in rxs[j][0]:
            x[n] -= 1
        for n in rxs[j][1]:
            x[n] += 1
        times.append(t)
        for n in names:
            hist[n].append(x[n])
    return np.array(times), {n: np.array(v) for n, v in hist.items()}
