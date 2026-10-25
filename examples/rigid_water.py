"""Rigid SPC water with Rattle constraints: same 8-molecule cluster idea as water_cluster.py, but O-H and H-H
distances are held fixed (3 constraints per molecule), which removes the fast O-H stretch.

Measured (2 ps NVE): energy drift 2.9e-4 at dt = 0.5 fs, 1.2e-3 at 1 fs and 4.8e-3 at 2 fs (clean 2nd order).
Bonds hold to 5e-13. So rigidity buys ~2.5x in dt here, not the textbook 10x. The report's
"timestep vs physics" row says why: close O...H electrostatic encounters (~4.5 fs) now limit dt, and
8 molecules in vacuum have a small energy scale. Constraints are solved in pure Python, so this is slow.

SPC: Berendsen, Postma, van Gunsteren & Hermans, in Intermolecular Forces (1981): r_OH = 1 A, H-O-H = 109.47 deg.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import Coulomb, LennardJones, Q, Rattle, Simulation, System

KCAL = 4184 / 6.02214076e23  # J per kcal/mol
r0, th0 = 1.0e-10, np.deg2rad(109.47)
d_hh = 2 * r0 * np.sin(th0 / 2)


def build():
    s = System(3)
    s.add_species("O", Q(15.9994, "u"), charge=Q(-0.82, "e"))
    s.add_species("H", Q(1.008, "u"), charge=Q(0.41, "e"))
    rng = np.random.default_rng(1)
    cons, excl = [], []
    for c in np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * 3.1e-10:
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        o = s.add("O", [c])[0]
        h = s.add("H", [c + R @ [r0, 0, 0], c + R @ [r0 * np.cos(th0), r0 * np.sin(th0), 0]])
        cons += [(o, h[0], r0), (o, h[1], r0), (h[0], h[1], d_hh)]
        excl += [(o, h[0]), (o, h[1]), (h[0], h[1])]
    s.maxwell_boltzmann(300.0, rng)
    # remove velocity components along the constraints before starting (RATTLE then keeps them out)
    w = {pid: n for n, pid in enumerate(s.ids)}
    inv = 1 / s.masses
    for _ in range(200):
        for a, b, d in cons:
            i, j = w[a], w[b]
            r = s.pos[i] - s.pos[j]
            k = r @ (s.vel[i] - s.vel[j]) / ((inv[i] + inv[j]) * d * d)
            s.vel[i] -= k * inv[i] * r
            s.vel[j] += k * inv[j] * r
    forces = [LennardJones({"O": (0.1553 * KCAL, 3.166e-10)}, cutoff=4), Coulomb(exclude=excl)]
    return s, forces, cons


if __name__ == "__main__":
    for dt in (0.5, 1.0, 2.0):
        s, forces, cons = build()
        sim = Simulation(s, forces, integrator=Rattle(cons), dt=Q(dt, "fs"))
        t0 = time.perf_counter()
        sim.run(duration=Q(2, "ps"), record_every=10, positions=False)
        c = {c.name: c for c in sim.accuracy_report().checks}
        print(f"dt = {dt} fs: energy drift {c['energy'].value:.2e} [{c['energy'].status}], "
              f"constraint violation {c['constraints'].value:.1e}, {time.perf_counter() - t0:.1f} s")
    print("(flexible SPC/Fw in water_cluster.py needs dt = 0.2 fs for a 4e-4 drift)")
    print(sim.accuracy_report())
