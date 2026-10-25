"""Time integrators. Symplectic ones keep energy error bounded over long runs."""
from __future__ import annotations

import numpy as np

from .units import k_B, si


class Integrator:
    order = 1
    symplectic = False
    stochastic = False
    thermostat = False  # exchanges energy with a heat bath

    def step(self, sim, dt):
        raise NotImplementedError


class VelocityVerlet(Integrator):
    """Kick-drift-kick leapfrog: 2nd order, symplectic, time reversible."""

    order = 2
    symplectic = True

    def step(self, sim, dt):
        s = sim.system
        s.vel += 0.5 * dt * sim.acc
        s.pos += dt * s.vel
        s.apply_boundaries()
        sim.update_forces()
        s.vel += 0.5 * dt * sim.acc


class Yoshida4(Integrator):
    """4th-order symplectic composition of three Verlet steps (Yoshida 1990)."""

    order = 4
    symplectic = True
    _w1 = 1 / (2 - 2 ** (1 / 3))
    _w0 = -(2 ** (1 / 3)) * _w1

    def __init__(self):
        self._vv = VelocityVerlet()

    def step(self, sim, dt):
        for w in (self._w1, self._w0, self._w1):
            self._vv.step(sim, w * dt)


class Langevin(Integrator):
    """BAOAB Langevin dynamics (Leimkuhler-Matthews): samples temperature T."""

    order = 2
    stochastic = True
    thermostat = True

    def __init__(self, temperature, friction):
        self.T = si(temperature, "K")
        self.gamma = si(friction, "1/s")

    def step(self, sim, dt):
        s = sim.system
        s.vel += 0.5 * dt * sim.acc
        s.pos += 0.5 * dt * s.vel
        c1 = np.exp(-self.gamma * dt)
        sig = np.sqrt((1 - c1 ** 2) * k_B.value * self.T / s.masses)[:, None]
        s.vel = c1 * s.vel + sig * sim.rng.normal(size=s.vel.shape)
        s.pos += 0.5 * dt * s.vel
        s.apply_boundaries()
        sim.update_forces()
        s.vel += 0.5 * dt * sim.acc


class Brownian(Integrator):
    """Overdamped Langevin (Euler-Maruyama) for diffusing molecules in a solvent.

    Uses each species' diffusion coefficient D: dx = D F/(kT) dt + sqrt(2 D dt) xi.
    """

    order = 1
    stochastic = True
    thermostat = True

    def __init__(self, temperature):
        self.T = si(temperature, "K")

    def step(self, sim, dt):
        s = sim.system
        D = np.array([sp.diffusion for sp in s.species])[s.sp][:, None]
        drift = D * sim.force / (k_B.value * self.T) * dt
        noise = np.sqrt(2 * D * dt) * sim.rng.normal(size=s.pos.shape)
        s.pos += drift + noise
        s.vel = (drift + noise) / dt
        s.apply_boundaries()
        sim.update_forces()
