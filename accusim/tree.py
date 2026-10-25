"""Barnes-Hut octree gravity: O(N log N), with a controllable, measurable error.

The tree walk is vectorised over all particles that visit a node, so Python
overhead scales with the number of nodes, not particles x nodes.
"""
from __future__ import annotations

import numpy as np

from .forces import Force
from .units import G as G_CONST
from .units import si


class _Node:
    __slots__ = ("center", "children", "com", "half", "idx", "mass")


class BarnesHut(Force):
    """Approximate Newtonian gravity. theta = 0 is exact direct summation."""

    exact = False

    def __init__(self, G=G_CONST, softening=0.0, theta=0.5, leaf_size=8):
        self.G = si(G, "m^3/kg/s^2")
        self.eps2 = si(softening, "m") ** 2
        self.theta = theta
        self.leaf_size = leaf_size

    def _build(self, pos, m, idx, center, half, depth=0):
        node = _Node()
        node.idx, node.center, node.half = idx, center, half
        node.mass = float(m[idx].sum())
        node.com = (m[idx, None] * pos[idx]).sum(0) / node.mass if node.mass > 0 else center
        node.children = []
        if len(idx) > self.leaf_size and depth < 48:
            octant = (pos[idx] > center).astype(int) @ (1 << np.arange(pos.shape[1]))
            for o in range(2 ** pos.shape[1]):
                sub = idx[octant == o]
                if len(sub):
                    sign = ((o >> np.arange(pos.shape[1])) & 1) * 2 - 1
                    node.children.append(self._build(pos, m, sub, center + sign * half / 2, half / 2, depth + 1))
        return node

    def compute(self, s):
        pos, m = s.pos, s.masses
        acc = np.zeros_like(pos)
        phi = np.zeros(s.n)
        if s.n < 2:
            return acc, 0.0
        lo, hi = pos.min(0), pos.max(0)
        half = max(float(np.max(hi - lo)) / 2 * (1 + 1e-9), 1e-300)
        root = self._build(pos, m, np.arange(s.n), (lo + hi) / 2, half)
        stack = [(root, np.arange(s.n))]
        th2 = self.theta ** 2
        while stack:
            node, tgt = stack.pop()
            if not node.children:
                src = node.idx
                d = pos[src][None, :, :] - pos[tgt][:, None, :]
                r2 = np.einsum("ijk,ijk->ij", d, d) + self.eps2
                same = tgt[:, None] == src[None, :]
                inv = np.where(same, 0.0, 1 / np.sqrt(np.where(same, 1.0, r2)))
                acc[tgt] += self.G * np.einsum("ij,ijk->ik", m[src][None, :] * inv ** 3, d)
                phi[tgt] -= self.G * (inv * m[src][None, :]).sum(1)
                continue
            d = node.com - pos[tgt]
            r2 = np.einsum("ij,ij->i", d, d) + self.eps2
            outside = np.any(np.abs(pos[tgt] - node.center) > node.half, axis=1)
            accept = outside & ((2 * node.half) ** 2 < th2 * r2)
            if np.any(accept):
                a, ra = tgt[accept], r2[accept]
                acc[a] += self.G * node.mass * d[accept] / ra[:, None] ** 1.5
                phi[a] -= self.G * node.mass / np.sqrt(ra)
            rest = tgt[~accept]
            if len(rest):
                stack.extend((c, rest) for c in node.children)
        return m[:, None] * acc, 0.5 * float(np.sum(m * phi))


def direct_subset(force, s, sub):
    """Exact gravitational force on particles `sub` from all particles (for error estimates)."""
    d = s.pos[None, :, :] - s.pos[sub][:, None, :]
    r2 = np.einsum("ijk,ijk->ij", d, d) + force.eps2
    same = sub[:, None] == np.arange(s.n)[None, :]
    inv = np.where(same, 0.0, 1 / np.sqrt(np.where(same, 1.0, r2)))
    return force.G * s.masses[sub, None] * np.einsum("ij,ijk->ik", s.masses[None, :] * inv ** 3, d)
