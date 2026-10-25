"""A + B <-> C in a gas: particle simulation vs exact stochastic (Gillespie) kinetics.

Particles move ballistically in a periodic box and react on contact (Doi model).
In the reaction-limited regime the rate constant is k = lambda * (4/3) pi R^3,
so the particle simulation should reproduce well-mixed mass-action kinetics.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import Q, Reaction, Simulation, System, gillespie

L, NA0, NB0 = 2e-7, 300, 300
LAM, R = 2e8, 4e-9
K_DECAY = 2e6


def build(seed=0):
    s = System(3, box=L, periodic=True)
    s.add_species("A", Q(20, "u"))
    s.add_species("B", Q(30, "u"))
    s.add_species("C", Q(50, "u"))
    rng = np.random.default_rng(seed)
    s.add("A", rng.uniform(0, L, (NA0, 3)))
    s.add("B", rng.uniform(0, L, (NB0, 3)))
    s.maxwell_boltzmann(Q(300, "K"), rng)
    return s


if __name__ == "__main__":
    s = build()
    rx = [Reaction(["A", "B"], "C", rate=LAM, radius=R, energy=Q(0.5, "eV")),
          Reaction("C", ["A", "B"], rate=K_DECAY, energy=Q(-0.5, "eV"))]
    sim = Simulation(s, reactions=rx, dt=Q(20, "ps"), seed=1)
    T = 2e-6
    tr = sim.run(duration=T, record_every=50)
    k_on = LAM * 4 / 3 * np.pi * R ** 3
    times, hist = gillespie({"A": NA0, "B": NB0}, [(("A", "B"), ("C",), k_on), (("C",), ("A", "B"), K_DECAY)],
                            L ** 3, T, rng=np.random.default_rng(2))
    print(f"particles: C = {tr.count('C')[-1]} after {T * 1e6:.1f} us; Gillespie (well-mixed exact): C = {hist['C'][-1]}")
    print(f"reaction events: {sim.reactions.events}")
    print(sim.accuracy_report())
    if "--gif" in sys.argv:
        from accusim.viz import animate
        os.makedirs("out", exist_ok=True)
        animate(tr, s, "out/chemistry.gif", unit=1e-9, unit_label="nm", title="A + B <-> C")
