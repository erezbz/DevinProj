"""Biology: Michaelis-Menten enzyme kinetics with diffusing molecules (Brownian dynamics).

E + S -> ES -> E + P, with E + S <- ES as well. Molecules diffuse with real-ish diffusion coefficients
and react on contact. The particle result is compared with exact well-mixed Gillespie kinetics.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import Brownian, Q, Reaction, Simulation, System, gillespie  # noqa: E402

L = 1e-6
NE, NS = 50, 500
LAM, R = 1e4, 21.544e-9
K_OFF, K_CAT = 50.0, 100.0


def build(seed=0):
    s = System(3, box=L, periodic=True)
    s.add_species("E", Q(50, "kDa"), diffusion=Q(60, "um^2/s"))
    s.add_species("S", Q(180, "Da"), diffusion=Q(500, "um^2/s"))
    s.add_species("ES", Q(50180, "Da"), diffusion=Q(60, "um^2/s"))
    s.add_species("P", Q(180, "Da"), diffusion=Q(500, "um^2/s"))
    rng = np.random.default_rng(seed)
    s.add("E", rng.uniform(0, L, (NE, 3)))
    s.add("S", rng.uniform(0, L, (NS, 3)))
    return s


if __name__ == "__main__":
    s = build()
    rx = [Reaction(["E", "S"], "ES", rate=LAM, radius=R),
          Reaction("ES", ["E", "S"], rate=K_OFF, separation=R),
          Reaction("ES", ["E", "P"], rate=K_CAT, separation=R)]
    sim = Simulation(s, reactions=rx, integrator=Brownian(Q(310, "K")), dt=Q(10, "us"), seed=1)
    T = 0.1
    tr = sim.run(duration=T, record_every=50, positions=False)
    k_on = LAM * 4 / 3 * np.pi * R ** 3
    print(f"diffusion-limited ceiling 4 pi D R = {4 * np.pi * 560e-12 * R:.2e} m^3/s, k_on used = {k_on:.2e} m^3/s")
    runs = [gillespie({"E": NE, "S": NS}, [(("E", "S"), ("ES",), k_on), (("ES",), ("E", "S"), K_OFF),
                                          (("ES",), ("E", "P"), K_CAT)], L ** 3, T,
                      rng=np.random.default_rng(k))[1]["P"][-1] for k in range(10)]
    print(f"products after {T * 1e3:.0f} ms: particles P = {tr.count('P')[-1]}, "
          f"Gillespie P = {np.mean(runs):.1f} +- {np.std(runs):.1f}")
    print(sim.accuracy_report())
