"""Simulation driver: ties system, forces, reactions and integrator together."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .integrators import VelocityVerlet
from .reactions import ReactionEngine
from .units import c as C_LIGHT
from .units import si


@dataclass
class Trajectory:
    t: list = field(default_factory=list)
    kinetic: list = field(default_factory=list)
    potential: list = field(default_factory=list)
    internal: list = field(default_factory=list)
    released: list = field(default_factory=list)
    momentum: list = field(default_factory=list)
    angular_momentum: list = field(default_factory=list)
    mass: list = field(default_factory=list)
    charge: list = field(default_factory=list)
    counts: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    species: list = field(default_factory=list)

    @property
    def total_energy(self) -> np.ndarray:
        """Mechanical + internal energy minus energy liberated from rest mass/bonds."""
        return (np.array(self.kinetic) + np.array(self.potential) + np.array(self.internal)
                - np.array(self.released))

    def count(self, name) -> np.ndarray:
        return np.array([c.get(name, 0) for c in self.counts])


class Simulation:
    def __init__(self, system, forces=(), reactions=(), integrator=None, dt=None, seed=None):
        self.system = system
        self.forces = list(forces)
        self.integrator = integrator or VelocityVerlet()
        self.dt = None if dt is None else si(dt, "s")
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.reactions = ReactionEngine(reactions, system) if reactions else None
        self.time = 0.0
        self.steps = 0
        self.trajectory = Trajectory()
        self.update_forces()
        self._m0 = self._mass_energy()
        self._initial = None

    def potential(self):
        s = self.system
        f = np.zeros_like(s.pos)
        u = 0.0
        for force in self.forces:
            fi, ui = force.compute(s)
            f += fi
            u += ui
        return f, u

    def update_forces(self):
        self.force, self.pe = self.potential()
        m = self.system.masses
        self.acc = self.force / m[:, None] if len(m) else self.force

    def _mass_energy(self):
        released = self.reactions.released_rest if self.reactions else 0.0
        return float(np.sum(self.system.masses)) + released / C_LIGHT.value ** 2

    def record(self, positions=True):
        s, tr = self.system, self.trajectory
        tr.t.append(self.time)
        tr.kinetic.append(s.kinetic_energy())
        tr.potential.append(self.pe)
        tr.internal.append((self.reactions.internal if self.reactions else 0.0) + float(np.sum(s.eint)))
        tr.released.append(self.reactions.released if self.reactions else 0.0)
        tr.momentum.append(s.momentum())
        tr.angular_momentum.append(s.angular_momentum() if s.dim in (2, 3) else np.zeros(1))
        tr.mass.append(self._mass_energy())
        tr.charge.append(float(np.sum(s.charges)))
        tr.counts.append(s.counts())
        if positions:
            tr.positions.append(s.pos.copy())
            tr.species.append(s.sp.copy())

    def run(self, duration=None, steps=None, record_every=1, positions=True):
        if self.dt is None:
            raise ValueError("set dt (or use accusim.advisor.suggest_dt)")
        if steps is None:
            steps = int(round(si(duration, "s") / self.dt))
        if not self.trajectory.t:
            if self._initial is None:
                self._initial = self.clone()
            self.record(positions)
        for k in range(1, steps + 1):
            self.integrator.step(self, self.dt)
            if self.reactions:
                self.reactions.step(self, self.dt)
            self.time += self.dt
            self.steps += 1
            if k % record_every == 0 or k == steps:
                self.record(positions)
        return self.trajectory

    def clone(self, dt=None):
        """Fresh simulation from the current state (used for convergence tests)."""
        import copy
        sim = Simulation.__new__(Simulation)
        sim.__dict__.update(self.__dict__)
        sim.system = self.system.copy()
        sim.forces = list(self.forces)
        sim.integrator = copy.deepcopy(self.integrator)
        sim.rng = np.random.default_rng(self.seed)
        sim.reactions = copy.deepcopy(self.reactions)
        sim.trajectory = Trajectory()
        sim.dt = self.dt if dt is None else dt
        sim.update_forces()
        return sim

    def accuracy_report(self, convergence_duration=None):
        from .accuracy import AccuracyReport
        return AccuracyReport.build(self, convergence_duration)
