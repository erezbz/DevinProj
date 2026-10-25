"""Particle state: species table plus flat numpy arrays (SI units)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .units import si


@dataclass(frozen=True)
class Species:
    name: str
    mass: float
    charge: float = 0.0
    radius: float = 0.0
    diffusion: float = 0.0


class System:
    """A set of particles in `dim` dimensions.

    box: edge lengths (SI); periodic=True wraps with minimum-image distances,
    periodic=False with a box reflects particles elastically off the walls.
    """

    def __init__(self, dim: int = 3, box=None, periodic: bool = False):
        self.dim = dim
        self.box = None if box is None else np.broadcast_to(si(box, "m"), (dim,)).astype(float).copy()
        self.periodic = periodic and self.box is not None
        self.species: list[Species] = []
        self._index: dict[str, int] = {}
        self.pos = np.zeros((0, dim))
        self.vel = np.zeros((0, dim))
        self.sp = np.zeros(0, dtype=int)
        self.ids = np.zeros(0, dtype=int)
        self.eint = np.zeros(0)  # internal (excitation/heat) energy per particle, J
        self._next_id = 0

    def add_species(self, name, mass, charge=0.0, radius=0.0, diffusion=0.0) -> Species:
        if name in self._index:
            raise ValueError(f"species '{name}' already defined")
        s = Species(name, si(mass, "kg"), si(charge, "C"), si(radius, "m"), si(diffusion, "m^2/s"))
        if s.mass <= 0:
            raise ValueError("mass must be positive")
        self._index[name] = len(self.species)
        self.species.append(s)
        return s

    def species_index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"unknown species '{name}'") from None

    def add(self, species: str, pos, vel=None, eint=0.0) -> np.ndarray:
        """Add one particle (pos shape (dim,)) or many (shape (n, dim)). Returns ids."""
        p = np.atleast_2d(si(pos, "m")).astype(float)
        v = np.zeros_like(p) if vel is None else np.broadcast_to(np.atleast_2d(si(vel, "m/s")), p.shape).astype(float)
        if p.shape[1] != self.dim:
            raise ValueError(f"positions must have {self.dim} components")
        n = len(p)
        new_ids = np.arange(self._next_id, self._next_id + n)
        self._next_id += n
        self.pos = np.vstack([self.pos, p])
        self.vel = np.vstack([self.vel, v])
        self.sp = np.concatenate([self.sp, np.full(n, self.species_index(species))])
        self.ids = np.concatenate([self.ids, new_ids])
        self.eint = np.concatenate([self.eint, np.broadcast_to(np.asarray(eint, float), (n,))])
        return new_ids

    def remove(self, mask: np.ndarray):
        keep = ~mask
        self.pos, self.vel, self.sp, self.ids = self.pos[keep], self.vel[keep], self.sp[keep], self.ids[keep]
        self.eint = self.eint[keep]

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def masses(self) -> np.ndarray:
        return np.array([s.mass for s in self.species])[self.sp] if self.n else np.zeros(0)

    @property
    def charges(self) -> np.ndarray:
        return np.array([s.charge for s in self.species])[self.sp] if self.n else np.zeros(0)

    def count(self, name: str) -> int:
        return int(np.sum(self.sp == self.species_index(name)))

    def counts(self) -> dict[str, int]:
        return {s.name: int(np.sum(self.sp == i)) for i, s in enumerate(self.species)}

    def displacements(self, idx_a=None, idx_b=None) -> np.ndarray:
        """r_j - r_i for all pairs, minimum image if periodic. Shape (na, nb, dim)."""
        a = self.pos if idx_a is None else self.pos[idx_a]
        b = self.pos if idx_b is None else self.pos[idx_b]
        d = b[None, :, :] - a[:, None, :]
        if self.periodic:
            d -= self.box * np.round(d / self.box)
        return d

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.masses[:, None] * self.vel ** 2))

    def momentum(self) -> np.ndarray:
        return np.sum(self.masses[:, None] * self.vel, axis=0)

    def angular_momentum(self) -> np.ndarray:
        L = np.cross(self.pos, self.masses[:, None] * self.vel)
        return np.atleast_1d(np.sum(L, axis=0))

    def center_of_mass_velocity(self) -> np.ndarray:
        return self.momentum() / np.sum(self.masses)

    def temperature(self) -> float:
        from .units import k_B
        dof = max(self.dim * self.n - self.dim, 1)
        return 2 * self.kinetic_energy() / (dof * k_B.value)

    def apply_boundaries(self):
        if self.box is None:
            return
        if self.periodic:
            self.pos %= self.box
            return
        for _ in range(4):
            low = self.pos < 0
            high = self.pos > self.box
            if not (low.any() or high.any()):
                break
            self.pos = np.where(low, -self.pos, self.pos)
            self.pos = np.where(high, 2 * self.box - self.pos, self.pos)
            self.vel = np.where(low | high, -self.vel, self.vel)

    def copy(self) -> "System":
        other = System.__new__(System)
        other.__dict__.update(self.__dict__)
        for k in ("pos", "vel", "sp", "ids", "eint"):
            setattr(other, k, getattr(self, k).copy())
        other.species = list(self.species)
        other._index = dict(self._index)
        other.box = None if self.box is None else self.box.copy()
        return other

    def maxwell_boltzmann(self, temperature, rng=None, zero_momentum=True):
        """Draw velocities at `temperature` (K) and remove net momentum."""
        from .units import k_B
        rng = rng or np.random.default_rng()
        T = si(temperature, "K")
        sigma = np.sqrt(k_B.value * T / self.masses)[:, None]
        self.vel = rng.normal(size=self.pos.shape) * sigma
        if zero_momentum and self.n > 1:
            self.vel -= self.center_of_mass_velocity()
