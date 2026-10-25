import numpy as np
import pytest

from accusim import (G, Gravity, HarmonicBond, Q, Simulation, System, VelocityVerlet, Yoshida4,
                     suggest_dt)


def kepler(integrator, steps_per_orbit=200, e=0.0):
    s = System(3)
    s.add_species("Sun", Q(1, "Msun"))
    s.add_species("Earth", Q(1, "Mearth"))
    GM = G.value * (s.species[0].mass + s.species[1].mass)
    a = Q(1, "au").value
    r_p = a * (1 - e)
    v_p = np.sqrt(GM * (1 + e) / r_p)
    mu = s.species[1].mass / (s.species[0].mass + s.species[1].mass)
    s.add("Sun", [-mu * r_p, 0, 0], [0, -mu * v_p, 0])
    s.add("Earth", [(1 - mu) * r_p, 0, 0], [0, (1 - mu) * v_p, 0])
    period = 2 * np.pi * np.sqrt(a ** 3 / GM)
    return Simulation(s, [Gravity()], integrator=integrator, dt=period / steps_per_orbit), period


@pytest.mark.parametrize("integ,tol", [(VelocityVerlet(), 2e-3), (Yoshida4(), 1e-6)])
def test_circular_orbit_returns_home_after_one_period(integ, tol):
    sim, period = kepler(integ, steps_per_orbit=400)
    start = sim.system.pos[1].copy()
    sim.run(steps=400)
    assert np.linalg.norm(sim.system.pos[1] - start) / np.linalg.norm(start) < tol


def test_eccentric_orbit_energy_bounded_over_100_orbits():
    sim, period = kepler(Yoshida4(), steps_per_orbit=500, e=0.5)
    sim.run(steps=500 * 100, record_every=50, positions=False)
    rep = sim.accuracy_report()
    assert rep["energy"].status == "PASS"
    assert rep["angular momentum"].status == "PASS"
    assert rep["momentum"].status == "PASS"


@pytest.mark.parametrize("integ", [VelocityVerlet(), Yoshida4()])
def test_report_measures_the_integrators_true_order(integ):
    sim, period = kepler(integ, steps_per_orbit=100, e=0.3)
    sim.run(steps=100, positions=False)
    chk = sim.accuracy_report(convergence_duration=period)["timestep error"]
    observed = float(chk.detail.split("observed order ")[1].split("/")[0])
    assert observed == pytest.approx(integ.order, abs=0.4)


def test_report_flags_a_bad_timestep():
    sim, period = kepler(VelocityVerlet(), steps_per_orbit=8, e=0.9)
    sim.run(steps=80, positions=False)
    assert sim.accuracy_report()["energy"].status in ("WARN", "FAIL")


def test_harmonic_oscillator_matches_exact_solution():
    s = System(1)
    s.add_species("m", Q(1, "kg"))
    s.add_species("wall", Q(1e12, "kg"))
    s.add("wall", [0.0])
    s.add("m", [1.1])
    k, r0 = 4.0, 1.0
    sim = Simulation(s, [HarmonicBond([(0, 1)], k=k, r0=r0)], integrator=Yoshida4(), dt=1e-3)
    sim.run(duration=Q(5, "s"), positions=False)
    w = np.sqrt(k / 1.0)
    assert s.pos[1, 0] == pytest.approx(r0 + 0.1 * np.cos(w * 5), abs=1e-6)


def test_advisor_suggests_fraction_of_orbit():
    sim, period = kepler(VelocityVerlet())
    dt = suggest_dt(sim)
    assert period / 1000 < dt < period / 50
