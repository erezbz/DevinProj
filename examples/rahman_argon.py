"""Reproduce Rahman (1964), the first molecular-dynamics simulation of a real liquid.

A. Rahman, "Correlations in the Motion of Atoms in Liquid Argon", Phys. Rev. 136, A405 (1964):
864 argon atoms, Lennard-Jones eps/k = 120 K, sigma = 3.4 A, density 1.374 g/cm^3, T = 94.4 K.
He found the diffusion coefficient D = 2.43e-5 cm^2/s (experiment: 2.43e-5) and the first g(r) peak near 3.7 A.
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import Langevin, LennardJones, Q, Simulation, System, VelocityVerlet, k_B
from accusim.analysis import msd, rdf

t0 = time.perf_counter()
m = Q(39.948, "u").value
n_cells = 6
rho = 1374.0  # kg/m^3
L = (4 * n_cells ** 3 * m / rho) ** (1 / 3)
base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) + 0.25
pts = np.array([b + [i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells) for b in base])
s = System(3, box=L, periodic=True)
s.add_species("Ar", m)
s.add("Ar", pts * L / n_cells)
rng = np.random.default_rng(64)
s.maxwell_boltzmann(94.4, rng)
lj = LennardJones({"Ar": (120 * k_B.value, 3.4e-10)}, cutoff=2.5)

melt = Simulation(s, [lj], integrator=Langevin(94.4, Q(2, "1/ps")), dt=Q(10, "fs"), seed=1)
melt.run(duration=Q(10, "ps"), positions=False)
print(f"{s.n} atoms, box {L * 1e10:.2f} A, melted and thermalised: T = {s.temperature():.1f} K")

sim = Simulation(s, [lj], integrator=VelocityVerlet(), dt=Q(10, "fs"))
prod_ps = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
tr = sim.run(duration=Q(prod_ps, "ps"), record_every=5)
# D from independent 10 ps blocks (fit the linear part, t > 2 ps), reported as mean +/- standard error
per_block = int(round(Q(10, "ps").value / (5 * sim.dt)))
Ds = []
for b0 in range(0, len(tr.t) - per_block, per_block):
    blk = SimpleNamespace(t=tr.t[b0:b0 + per_block + 1], positions=tr.positions[b0:b0 + per_block + 1])
    t, r2 = msd(blk, box=L)
    late = t > Q(2, "ps").value
    Ds.append(np.polyfit(t[late], r2[late], 1)[0] / 6)
D, dD = np.mean(Ds), (np.std(Ds, ddof=1) / np.sqrt(len(Ds)) if len(Ds) > 1 else float("nan"))
r, g = rdf(s, 0.49 * L, bins=120)
print(f"NVE production {prod_ps:g} ps: mean T = {np.mean(2 * np.array(tr.kinetic) / (3 * (s.n - 1) * k_B.value)):.1f} K")
print(f"diffusion D = ({D * 1e9:.2f} +/- {dD * 1e9:.2f})e-5 cm^2/s from {len(Ds)} blocks   (Rahman 1964: 2.43e-5, experiment: 2.43e-5)")
print(f"g(r) first peak at {r[np.argmax(g)] * 1e10:.2f} A, height {g.max():.2f}   (Rahman: ~3.7 A)")
print(sim.accuracy_report())
print(f"wall time {time.perf_counter() - t0:.0f} s")
