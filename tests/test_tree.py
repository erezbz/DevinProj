import os
import sys

import numpy as np
import pytest

from accusim import BarnesHut, Gravity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from star_cluster import plummer  # noqa: E402


def _err(theta, n=1500):
    s, a, M = plummer(n=n, seed=1)
    F0, U0 = Gravity(softening=0.01 * a).compute(s)
    F1, U1 = BarnesHut(softening=0.01 * a, theta=theta).compute(s)
    ferr = np.sqrt(np.mean(np.sum((F1 - F0) ** 2, 1)) / np.mean(np.sum(F0 ** 2, 1)))
    return ferr, abs(U1 - U0) / abs(U0)


def test_theta_zero_is_exact():
    ferr, uerr = _err(0.0, n=300)
    assert ferr < 1e-12 and uerr < 1e-12


@pytest.mark.parametrize("theta,tol", [(0.3, 2e-3), (0.6, 1e-2)])
def test_force_error_is_small_and_controlled_by_theta(theta, tol):
    ferr, uerr = _err(theta)
    assert ferr < tol and uerr < tol


def test_report_measures_tree_force_error():
    from accusim import Simulation, Yoshida4
    s, a, M = plummer(n=600, seed=2)
    sim = Simulation(s, [BarnesHut(softening=0.01 * a, theta=0.7)], integrator=Yoshida4(), dt=1e10)
    sim.run(steps=3, positions=False)
    chk = sim.accuracy_report()["force approximation"]
    ferr, _ = _err(0.7, n=600)
    assert chk.status in ("PASS", "WARN") and 0 < chk.value < 2e-2


def test_advisor_scales_to_large_tree_runs():
    from accusim import Simulation
    from accusim.advisor import timescales
    s, a, M = plummer(n=10000, seed=3)
    ts = timescales(Simulation(s, [BarnesHut(softening=0.01 * a)], dt=1e10))
    assert "gravity (orbital)" in ts and ts["closest approach / speed"] > 0
