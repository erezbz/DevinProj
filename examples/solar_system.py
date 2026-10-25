"""Inner solar system + Jupiter for 20 years with real masses and orbits.

Checks Earth's simulated year against 365.256 days and prints the accuracy report.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import G, Gravity, Q, Simulation, System, Yoshida4, suggest_dt

# name, mass, semi-major axis (au), eccentricity  (approximate J2000 elements)
BODIES = [
    ("Mercury", Q(0.33011e24, "kg"), 0.387098, 0.205630),
    ("Venus", Q(4.8675e24, "kg"), 0.723332, 0.006772),
    ("Earth", Q(1, "Mearth"), 1.000001, 0.016709),
    ("Mars", Q(0.64171e24, "kg"), 1.523679, 0.0934),
    ("Jupiter", Q(1, "Mjup"), 5.2044, 0.0489),
]


def build():
    s = System(3)
    s.add_species("Sun", Q(1, "Msun"))
    s.add("Sun", [0, 0, 0])
    GM = G.value * s.species[0].mass
    rng = np.random.default_rng(42)
    for name, m, a, e in BODIES:
        s.add_species(name, m)
        a_m = Q(a, "au").value
        r = a_m * (1 - e)
        v = np.sqrt(GM * (1 + e) / r)  # perihelion speed
        th = rng.uniform(0, 2 * np.pi)
        s.add(name, [r * np.cos(th), r * np.sin(th), 0], [-v * np.sin(th), v * np.cos(th), 0])
    s.vel -= s.center_of_mass_velocity()
    return s


def earth_year(tr, s):
    i = s.species_index("Earth")
    idx = int(np.flatnonzero(s.sp == i)[0])
    p = np.array([x[idx] - x[0] for x in tr.positions])
    ang = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    t = np.array(tr.t)
    return 2 * np.pi / np.polyfit(t, ang, 1)[0]


if __name__ == "__main__":
    s = build()
    sim = Simulation(s, [Gravity()], integrator=Yoshida4())
    sim.dt = suggest_dt(sim, steps_per_timescale=60)
    print(f"advisor chose dt = {sim.dt / 86400:.3f} days")
    tr = sim.run(duration=Q(20, "yr"), record_every=20)
    print(f"Earth's mean sidereal year: {earth_year(tr, s) / 86400:.2f} days (real: 365.26 days, perturbed by planets)")
    print(sim.accuracy_report(convergence_duration=Q(1, "yr")))
    if "--gif" in sys.argv:
        from accusim.viz import animate
        os.makedirs("out", exist_ok=True)
        animate(tr, s, "out/solar_system.gif", unit=Q(1, "au").value, unit_label="au", title="Solar system")
