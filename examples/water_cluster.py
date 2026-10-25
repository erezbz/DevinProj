"""Flexible water (SPC/Fw parameters): bonds + angles + Lennard-Jones + Coulomb with
intramolecular exclusions. Eight molecules in vacuum, NVE, 2 ps.

SPC/Fw: Wu, Tepper & Voth, J. Chem. Phys. 124, 024503 (2006).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import Coulomb, HarmonicAngle, HarmonicBond, LennardJones, Q, Simulation, System, VelocityVerlet  # noqa: E402

KCAL = 4184 / 6.02214076e23  # J per kcal/mol
s = System(3)
s.add_species("O", Q(15.9994, "u"), charge=Q(-0.82, "e"))
s.add_species("H", Q(1.008, "u"), charge=Q(0.41, "e"))
r0, th0 = 1.012e-10, np.deg2rad(113.24)
rng = np.random.default_rng(1)
bonds, angles, excl = [], [], []
for c in np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * 3.1e-10:
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    o = s.add("O", [c])[0]
    h = s.add("H", [c + R @ [r0, 0, 0], c + R @ [r0 * np.cos(th0), r0 * np.sin(th0), 0]])
    bonds += [(o, h[0]), (o, h[1])]
    angles.append((h[0], o, h[1]))
    excl += [(o, h[0]), (o, h[1]), (h[0], h[1])]
s.maxwell_boltzmann(300.0, rng)

forces = [
    HarmonicBond(bonds, k=1059.162 * KCAL / 1e-20, r0=r0),
    HarmonicAngle(angles, k=75.90 * KCAL, theta0=th0),
    LennardJones({"O": (0.1554 * KCAL, 3.165492e-10)}, cutoff=4),
    Coulomb(exclude=excl),
]
sim = Simulation(s, forces, integrator=VelocityVerlet(), dt=Q(0.2, "fs"))
lengths, angs = [], []
for _ in range(100):
    sim.run(steps=100, positions=False)
    w = {pid: n for n, pid in enumerate(s.ids)}
    lengths += [np.linalg.norm(s.pos[w[b]] - s.pos[w[a]]) for a, b in bonds]
    for a, b, cc in angles:
        u, v = s.pos[w[a]] - s.pos[w[b]], s.pos[w[cc]] - s.pos[w[b]]
        angs.append(np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))

print("8 flexible SPC/Fw waters, 2 ps NVE at dt = 0.2 fs")
print(f"mean O-H length {np.mean(lengths) * 1e10:.4f} A (SPC/Fw r0 = 1.012 A; thermal stretching makes it slightly longer)")
print(f"mean H-O-H angle {np.mean(angs):.2f} deg (theta0 = 113.24 deg; H-H Coulomb is excluded)")
print(sim.accuracy_report())
