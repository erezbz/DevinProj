"""Molten NaCl at 1100 K with cutoff electrostatics, checked against Ewald.

n^3 ions (default 216) at the experimental melt density (1.556 g/cm^3), Joung-Cheatham Lennard-Jones
parameters, DampedShiftedCoulomb (alpha = 0.2 / A, rcut = min(12 A, 0.48 box)). Langevin melting, then NVE.

Finding: the report's "DSF vs Ewald" row shows ~12% RMS force error both for 216 ions (rcut = 9 A, scanning
alpha 0.1-0.3 never got below 9%) and for 512 ions with the published rcut = 12 A. That matches the
literature's force correlation R^2 ~ 0.99 for DSF (sqrt(0.01) = 10%), so it is graded WARN: fine for
structure (Na-Cl peak 2.59 A), but use Ewald if forces must be exact. `python examples/molten_salt.py 8` runs 512 ions.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from accusim import Q, Simulation, System  # noqa: E402
from accusim.forces import DampedShiftedCoulomb, LennardJones  # noqa: E402
from accusim.integrators import Langevin, VelocityVerlet  # noqa: E402

N_A = 6.02214076e23
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6  # ions = n^3
mass_g = n ** 3 / 2 * (22.98977 + 35.453) / N_A
L = (mass_g / 1.556) ** (1 / 3) * 1e-2
s = System(3, box=L, periodic=True)
s.add_species("Na", Q(22.98977, "u"), charge=Q(1, "e"))
s.add_species("Cl", Q(35.453, "u"), charge=Q(-1, "e"))
g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
odd = g.sum(1) % 2 == 1
s.add("Na", (g[~odd] + 0.5) * L / n)
s.add("Cl", (g[odd] + 0.5) * L / n)
s.maxwell_boltzmann(1100, rng=np.random.default_rng(0))
forces = [LennardJones({"Na": (1.4754e3 / N_A, 2.159e-10), "Cl": (0.0128e3 / N_A, 4.830e-10)}, cutoff=1.9),
          DampedShiftedCoulomb(rcut=min(12e-10, 0.48 * L), alpha=Q(0.2, "1/A"))]
t0 = time.time()
lattice = s.pos.copy()
for T, ps in [(3000, 2), (1100, 3)]:  # melt hot (a perfect crystal superheats), then cool to 1100 K
    Simulation(s, forces, integrator=Langevin(T, Q(5, "1/ps")), dt=Q(1, "fs")).run(duration=Q(ps, "ps"), record_every=100)
moved = s.pos - lattice
moved -= s.box * np.round(moved / s.box)
print(f"rms displacement from the initial lattice {np.sqrt(np.mean(np.sum(moved ** 2, 1))) * 1e10:.1f} A (melted if >> 0.3 A)")
sim = Simulation(s, forces, integrator=VelocityVerlet(), dt=Q(1, "fs"))
tr = sim.run(duration=Q(2, "ps"), record_every=20)
na, cl = np.flatnonzero(s.sp == 0), np.flatnonzero(s.sp == 1)
d = s.pos[cl][None] - s.pos[na][:, None]
d -= s.box * np.round(d / s.box)
r = np.sqrt(np.einsum("ijk,ijk->ij", d, d)).ravel() * 1e10
h, e = np.histogram(r, bins=60, range=(1.5, 6.0))
h = h / (0.5 * (e[1:] + e[:-1])) ** 2  # shell volume, so the peak is g(r)'s, not r^2-weighted counts
print(f"box {L * 1e10:.2f} A, NVE mean T = {np.mean([2 * k / (3 * s.n * 1.380649e-23) for k in tr.kinetic]):.0f} K")
print(f"Na-Cl g(r) first peak near {0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]):.2f} A (experiment ~2.6-2.8 A)")
print(sim.accuracy_report())
print(f"wall time {time.time() - t0:.0f} s")
