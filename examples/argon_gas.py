"""Liquid-ish argon: 216 atoms with real Lennard-Jones parameters in a periodic box.

Starts at 120 K, runs 10 ps with Velocity Verlet, and reports energy conservation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import N_A, LennardJones, Q, Simulation, System, VelocityVerlet, suggest_dt

EPS = Q(0.996, "kJ/mol") / N_A   # Rahman 1964: eps/k_B = 120 K
SIGMA = Q(3.405, "A")


def build(n_side=6, density_g_cm3=1.0, T=120, seed=0):
    m = Q(39.948, "u")
    n = n_side ** 3
    L = ((n * m) / Q(density_g_cm3, "g/cm^3")) ** (1 / 3)
    s = System(3, box=L, periodic=True)
    s.add_species("Ar", m)
    g = (np.arange(n_side) + 0.5) / n_side
    pts = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3) * L.value
    s.add("Ar", pts)
    s.maxwell_boltzmann(Q(T, "K"), np.random.default_rng(seed))
    return s


if __name__ == "__main__":
    s = build()
    sim = Simulation(s, [LennardJones({"Ar": (EPS, SIGMA)})], integrator=VelocityVerlet())
    sim.dt = min(suggest_dt(sim, steps_per_timescale=200), Q(5, "fs").value)
    print(f"box {s.box[0] * 1e9:.3f} nm, dt = {sim.dt * 1e15:.2f} fs")
    tr = sim.run(duration=Q(10, "ps"), record_every=10)
    print(f"temperature: start {120:.0f} K -> end {s.temperature():.1f} K (equilibrating KE <-> PE)")
    print(sim.accuracy_report(convergence_duration=Q(0.5, "ps")))
    if "--gif" in sys.argv:
        from accusim.viz import animate
        os.makedirs("out", exist_ok=True)
        animate(tr, s, "out/argon.gif", unit=1e-9, unit_label="nm", title="Argon (LJ)")
