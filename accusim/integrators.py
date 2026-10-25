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


class NoseHoover(Integrator):
    """Nose-Hoover thermostat (Trotter splitting, Martyna-Tuckerman-Klein 1996): canonical sampling with a conserved
    extended energy E + Q xi^2/2 + N_f kT eta, which the accuracy report checks. `period` sets the thermostat mass."""

    order = 2
    thermostat = True

    def __init__(self, temperature, period):
        self.T = si(temperature, "K")
        self.tau = si(period, "s")
        self.xi = 0.0
        self.eta = 0.0
        self.Q = None

    @staticmethod
    def dof(s):
        return s.dim * max(s.n - 1, 1)

    def _half(self, s, dt):
        kT, nf = k_B.value * self.T, self.dof(s)
        if self.Q is None:
            self.Q = nf * kT * self.tau ** 2
        k2 = float(np.sum(s.masses[:, None] * s.vel ** 2))
        self.xi += 0.25 * dt * (k2 - nf * kT) / self.Q
        scale = np.exp(-0.5 * dt * self.xi)
        s.vel *= scale
        self.eta += 0.5 * dt * self.xi
        self.xi += 0.25 * dt * (k2 * scale ** 2 - nf * kT) / self.Q

    def step(self, sim, dt):
        s = sim.system
        self._half(s, dt)
        s.vel += 0.5 * dt * sim.acc
        s.pos += dt * s.vel
        s.apply_boundaries()
        sim.update_forces()
        s.vel += 0.5 * dt * sim.acc
        self._half(s, dt)

    def bath_energy(self, s):
        if self.Q is None:
            return 0.0
        return 0.5 * self.Q * self.xi ** 2 + self.dof(s) * k_B.value * self.T * self.eta


class Rattle(Integrator):
    """Velocity Verlet with rigid bond-length constraints: SHAKE on positions, RATTLE on velocities (Andersen 1983).

    constraints: [(id_a, id_b, length)] using stable particle IDs. Constraint forces do no work, so energy stays
    conserved; the accuracy report checks the worst relative bond-length violation of the run (`max_violation`).
    Constraints are solved one at a time (Gauss-Seidel), which also converges for coupled ones (e.g. rigid water).
    """

    order = 2
    symplectic = True

    def __init__(self, constraints, tol=1e-12, max_iter=1000):
        self.constraints = [(int(a), int(b), si(d, "m")) for a, b, d in constraints]
        self.tol = tol
        self.max_iter = max_iter
        self.max_violation = 0.0

    @staticmethod
    def _mi(s, d):
        return d - s.box * np.round(d / s.box) if s.periodic else d

    def step(self, sim, dt):
        s = sim.system
        where = {pid: n for n, pid in enumerate(s.ids)}
        cons = [(where[a], where[b], d) for a, b, d in self.constraints if a in where and b in where]
        inv = 1.0 / s.masses
        old = [self._mi(s, s.pos[a] - s.pos[b]) for a, b, _ in cons]
        s.vel += 0.5 * dt * sim.acc
        new = s.pos + dt * s.vel
        for _ in range(self.max_iter):
            done = True
            for (a, b, d), ro in zip(cons, old):
                r = self._mi(s, new[a] - new[b])
                diff = d * d - r @ r
                if abs(diff) > self.tol * d * d:
                    done = False
                    g = diff / (2 * (inv[a] + inv[b]) * (r @ ro))
                    new[a] += g * inv[a] * ro
                    new[b] -= g * inv[b] * ro
                    s.vel[a] += g * inv[a] * ro / dt
                    s.vel[b] -= g * inv[b] * ro / dt
            if done:
                break
        else:
            raise RuntimeError("SHAKE did not converge: reduce dt")
        s.pos = new
        s.apply_boundaries()
        sim.update_forces()
        s.vel += 0.5 * dt * sim.acc
        for _ in range(self.max_iter):
            done = True
            for a, b, d in cons:
                r = self._mi(s, s.pos[a] - s.pos[b])
                rv = r @ (s.vel[a] - s.vel[b])
                if abs(rv) * dt > self.tol * d * d:
                    done = False
                    k = rv / ((inv[a] + inv[b]) * d * d)
                    s.vel[a] -= k * inv[a] * r
                    s.vel[b] += k * inv[b] * r
            if done:
                break
        else:
            raise RuntimeError("RATTLE did not converge: reduce dt")
        for a, b, d in cons:
            r = self._mi(s, s.pos[a] - s.pos[b])
            self.max_violation = max(self.max_violation, abs(np.sqrt(r @ r) - d) / d)


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
