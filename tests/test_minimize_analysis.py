import numpy as np
import pytest

from accusim import HarmonicAngle, HarmonicBond, LennardJones, Q, Simulation, System
from accusim.analysis import rdf
from accusim.minimize import minimize


def test_fire_finds_lennard_jones_dimer_minimum():
    eps, sig = 1.65e-21, 3.4e-10
    s = System(3)
    s.add_species("Ar", Q(39.948, "u"))
    s.add("Ar", [[0, 0, 0], [4.5e-10, 0.3e-10, 0]])
    sim = Simulation(s, [LennardJones({"Ar": (eps, sig)}, cutoff=5)], dt=Q(5, "fs"))
    U, fmax, _ = minimize(sim)
    r = np.linalg.norm(s.pos[1] - s.pos[0])
    assert r == pytest.approx(2 ** (1 / 6) * sig, rel=1e-6)
    assert U == pytest.approx(-eps * (1 - 0) + 0, rel=2e-3)  # shifted LJ: -eps + |shift|, tiny at 5 sigma


def test_fire_relaxes_a_distorted_molecule_to_its_rest_geometry():
    s = System(3)
    s.add_species("O", Q(16, "u"))
    s.add_species("H", Q(1, "u"))
    o = s.add("O", [[0, 0, 0]])[0]
    h = s.add("H", [[1.3e-10, 0, 0], [0, 0.8e-10, 0.2e-10]])
    th0 = np.deg2rad(104.5)
    sim = Simulation(s, [HarmonicBond([(o, h[0]), (o, h[1])], k=500.0, r0=1e-10),
                         HarmonicAngle([(h[0], o, h[1])], k=1e-19, theta0=th0)], dt=Q(0.1, "fs"))
    minimize(sim)
    u, v = s.pos[1] - s.pos[0], s.pos[2] - s.pos[0]
    assert np.linalg.norm(u) == pytest.approx(1e-10, rel=1e-6)
    assert np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)) == pytest.approx(th0, rel=1e-6)


def test_rdf_of_ideal_gas_is_one():
    s = System(3, box=1.0, periodic=True)
    s.add_species("X", 1.0)
    s.add("X", np.random.default_rng(0).uniform(0, 1, (3000, 3)))
    r, g = rdf(s, 0.5, bins=20)
    assert np.allclose(g, 1.0, atol=0.06)


def test_rdf_of_fcc_crystal_peaks_at_nearest_neighbour_distance():
    a, n = 1.0, 4
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    pts = np.array([b + [i, j, k] for i in range(n) for j in range(n) for k in range(n) for b in base]) * a
    s = System(3, box=n * a, periodic=True)
    s.add_species("X", 1.0)
    s.add("X", pts)
    r, g = rdf(s, 1.5, bins=150)
    assert r[np.argmax(g)] == pytest.approx(a / np.sqrt(2), abs=0.01)
