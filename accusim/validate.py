"""python -m accusim.validate: run textbook benchmarks with known answers and print a scorecard.

The point of accusim is that it proves itself: each row compares the engine
against an exact or reference result that does not come from accusim.
"""
from __future__ import annotations

import time

import numpy as np

from .forces import Ewald, Gravity, HarmonicBond
from .integrators import Brownian, VelocityVerlet, Yoshida4
from .reactions import Reaction, gillespie
from .simulation import Simulation
from .system import System
from .units import G, Q


def kepler_period():
    s = System(3)
    s.add_species("Sun", Q(1, "Msun"))
    s.add_species("Earth", Q(1, "Mearth"))
    GM = G.value * (s.species[0].mass + s.species[1].mass)
    a = Q(1, "au").value
    mu = s.species[1].mass / (s.species[0].mass + s.species[1].mass)
    v = np.sqrt(GM / a)
    s.add("Sun", [-mu * a, 0, 0], [0, -mu * v, 0])
    s.add("Earth", [(1 - mu) * a, 0, 0], [0, (1 - mu) * v, 0])
    T = 2 * np.pi * np.sqrt(a ** 3 / GM)
    sim = Simulation(s, [Gravity()], integrator=Yoshida4(), dt=T / 1000)
    start = s.pos[1] - s.pos[0]
    sim.run(steps=1000, positions=False)
    end = s.pos[1] - s.pos[0]
    return "Circular orbit closes after Kepler period", 0.0, float(np.linalg.norm(end - start) / a), "rel. position", 1e-8


def oscillator():
    s = System(1)
    s.add_species("m", 1.0)
    s.add_species("wall", 1e15)
    s.add("wall", [0.0])
    s.add("m", [1.1])
    sim = Simulation(s, [HarmonicBond([(0, 1)], k=4.0, r0=1.0)], integrator=Yoshida4(), dt=1e-3)
    sim.run(steps=5000, positions=False)
    exact = 1.0 + 0.1 * np.cos(2.0 * 5.0)
    return "Harmonic oscillator x(t) = A cos(wt)", exact, float(s.pos[1, 0]), "m", 1e-6


def verlet_order():
    errs = []
    for n in (200, 400):
        s = System(1)
        s.add_species("m", 1.0)
        s.add_species("wall", 1e15)
        s.add("wall", [0.0])
        s.add("m", [1.1])
        sim = Simulation(s, [HarmonicBond([(0, 1)], k=4.0, r0=1.0)], integrator=VelocityVerlet(), dt=2.0 / n)
        sim.run(steps=n, positions=False)
        errs.append(abs(s.pos[1, 0] - (1.0 + 0.1 * np.cos(4.0))))
    return "Velocity Verlet convergence order", 2.0, float(np.log2(errs[0] / errs[1])), "order", 0.05


def madelung():
    s = System(3, box=2.0, periodic=True)
    s.add_species("Na", 1.0, charge=1.0)
    s.add_species("Cl", 1.0, charge=-1.0)
    for x in range(2):
        for y in range(2):
            for z in range(2):
                s.add("Na" if (x + y + z) % 2 == 0 else "Cl", [[x, y, z]])
    _, U = Ewald(k=1.0).compute(s)
    return "NaCl Madelung constant (Ewald)", 1.747564594633, float(-U / 4), "", 1e-7


def dt_fusion():
    s = System(3)
    s.add_species("D", Q(2.01410177812, "u"), charge=Q(1, "e"))
    s.add_species("T", Q(3.0160492779, "u"), charge=Q(1, "e"))
    s.add_species("He4", Q(4.00260325413, "u"), charge=Q(2, "e"))
    s.add_species("n", Q(1.00866491595, "u"))
    r = Reaction(["D", "T"], ["He4", "n"], rate=1, radius=1e-14)
    Simulation(s, reactions=[r])
    return "D-T fusion Q-value from mass defect", 17.5893, r.q_value / Q(1, "MeV").value, "MeV", 1e-4


def decay():
    s = System(3, box=1e-7, periodic=True)
    s.add_species("A", 1e-26)
    s.add_species("A*", 1e-26)
    s.add("A", np.random.default_rng(0).uniform(0, 1e-7, (20000, 3)))
    sim = Simulation(s, reactions=[Reaction("A", "A*", rate=1e9)], dt=1e-11, seed=1)
    sim.run(steps=100, positions=False)
    return "First-order decay N(t) = N0 exp(-kt)", 20000 * np.exp(-1.0), float(s.count("A")), "particles", 0.02


def mass_action():
    k, V, n0 = 1e-20, 1e-18, 1000
    rng = np.random.default_rng(3)
    finals = [gillespie({"A": n0, "B": n0}, [(("A", "B"), ("C",), k)], V, 0.5, rng=rng)[1]["A"][-1] for _ in range(20)]
    return "Gillespie A+B->C vs rate equation", n0 / (1 + k / V * n0 * 0.5), float(np.mean(finals)), "molecules", 0.03


def diffusion():
    s = System(3)
    D = 1e-9
    s.add_species("X", 1e-25, diffusion=D)
    s.add("X", np.zeros((20000, 3)))
    sim = Simulation(s, integrator=Brownian(300.0), dt=1e-9, seed=4)
    sim.run(steps=100, positions=False)
    return "Einstein diffusion <r^2> = 6Dt", 6 * D * 1e-7, float(np.mean(np.sum(s.pos ** 2, 1))), "m^2", 0.03


BENCHMARKS = [kepler_period, oscillator, verlet_order, madelung, dt_fusion, decay, mass_action, diffusion]


def main():
    print(f"{'benchmark':<44}{'expected':>14}{'accusim':>14}  {'rel.err':>9}  result  time")
    ok = True
    for bench in BENCHMARKS:
        t0 = time.perf_counter()
        name, exp, got, unit, tol = bench()
        err = abs(got) if exp == 0 else abs(got - exp) / abs(exp)
        good = err <= tol
        ok &= good
        print(f"{name:<44}{exp:>14.6g}{got:>14.6g}  {err:>9.1e}  {'PASS' if good else 'FAIL':<6}  "
              f"{time.perf_counter() - t0:.1f}s")
    print("all benchmarks passed" if ok else "SOME BENCHMARKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
