"""Astronomy: a Plummer-sphere star cluster of 400 solar-mass stars, run for 2 crossing times.

Checks the virial theorem (2K/|U| ~ 1 for an equilibrium cluster) and energy conservation.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import G, Gravity, Q, Simulation, System, Yoshida4


def plummer(n=400, a=Q(1, "pc").value, m=Q(1, "Msun").value, seed=0):
    """Aarseth-Henon-Wielen (1974) sampling of an equilibrium Plummer model."""
    rng = np.random.default_rng(seed)
    M = n * m
    r = a / np.sqrt(rng.uniform(1e-3, 0.99, n) ** (-2 / 3) - 1)

    def iso(k):
        u = rng.normal(size=(k, 3))
        return u / np.linalg.norm(u, axis=1)[:, None]

    pos = r[:, None] * iso(n)
    q = np.empty(n)
    for i in range(n):
        while True:
            x, y = rng.uniform(0, 1), rng.uniform(0, 0.1)
            if y < x ** 2 * (1 - x ** 2) ** 3.5:
                q[i] = x
                break
    vesc = np.sqrt(2 * G.value * M / np.sqrt(r ** 2 + a ** 2))
    vel = (q * vesc)[:, None] * iso(n)
    s = System(3)
    s.add_species("star", m)
    s.add("star", pos - pos.mean(0), vel - vel.mean(0))
    return s, a, M


if __name__ == "__main__":
    s, a, M = plummer()
    soft = 0.01 * a
    grav = Gravity(softening=soft)
    t_cross = np.sqrt(a ** 3 / (G.value * M))
    sim = Simulation(s, [grav], integrator=Yoshida4(), dt=t_cross / 400)
    _, U0 = grav.compute(s)
    print(f"crossing time {t_cross / Q(1, 'yr').value / 1e6:.2f} Myr, virial ratio 2K/|U| at t=0: "
          f"{2 * s.kinetic_energy() / abs(U0):.3f}")
    tr = sim.run(duration=2 * t_cross, record_every=20, positions="--gif" in sys.argv)
    _, U1 = grav.compute(s)
    print(f"virial ratio after 2 crossing times: {2 * s.kinetic_energy() / abs(U1):.3f}")
    print(sim.accuracy_report())
    if "--gif" in sys.argv:
        from accusim.viz import animate3d
        os.makedirs("out", exist_ok=True)
        animate3d(tr, s, "out/star_cluster.gif", unit=Q(1, "pc").value, unit_label="pc", title="Plummer cluster")
