"""Every conservative force must satisfy F = -grad U (checked numerically)."""
import numpy as np
import pytest

from accusim import Coulomb, Gravity, HarmonicBond, LennardJones, Q, System, UniformField


def _system(periodic=False):
    rng = np.random.default_rng(0)
    s = System(3, box=Q(2.0, "nm") if periodic else None, periodic=periodic)
    s.add_species("A", Q(40, "u"), charge=Q(1, "e"))
    s.add_species("B", Q(20, "u"), charge=Q(-1, "e"))
    pts = rng.uniform(0.2e-9, 1.8e-9, size=(6, 3))
    s.add("A", pts[:3])
    s.add("B", pts[3:])
    return s


FORCES = [
    lambda s: Gravity(),
    lambda s: Coulomb(),
    lambda s: LennardJones({"A": (Q(1.0, "kJ") / 6.022e23, Q(3.4, "A")), "B": (Q(0.5, "kJ") / 6.022e23, Q(2.8, "A"))}, cutoff=5),
    lambda s: HarmonicBond([(0, 3), (1, 4)], k=Q(300, "N/m"), r0=Q(1, "A")),
    lambda s: UniformField(g=[0, 0, -9.8], E=[1e5, 0, 0]),
]


@pytest.mark.parametrize("make", FORCES)
@pytest.mark.parametrize("periodic", [False, True])
def test_force_is_negative_gradient(make, periodic):
    s = _system(periodic)
    f = make(s)
    F, _ = f.compute(s)
    h = 1e-15
    for i in range(s.n):
        for k in range(3):
            s.pos[i, k] += h
            up = f.compute(s)[1]
            s.pos[i, k] -= 2 * h
            down = f.compute(s)[1]
            s.pos[i, k] += h
            num = -(up - down) / (2 * h)
            assert num == pytest.approx(F[i, k], rel=1e-4, abs=1e-6 * np.max(np.abs(F)) + 1e-300)


@pytest.mark.parametrize("make", FORCES[:4])
def test_pair_forces_obey_newtons_third_law(make):
    s = _system()
    F, _ = make(s).compute(s)
    assert np.allclose(F.sum(axis=0), 0, atol=1e-10 * np.max(np.abs(F)))


def test_lj_minimum_at_two_to_the_sixth_sigma():
    s = System(3)
    s.add_species("Ar", Q(39.948, "u"))
    sigma = 3.4e-10
    s.add("Ar", [[0, 0, 0], [2 ** (1 / 6) * sigma, 0, 0]])
    F, _ = LennardJones({"Ar": (1e-21, sigma)}, cutoff=10).compute(s)
    assert np.allclose(F, 0, atol=1e-20)


@pytest.mark.parametrize("periodic", [False, True])
def test_cell_list_matches_all_pairs(periodic):
    rng = np.random.default_rng(5)
    s = System(3, box=Q(4, "nm"), periodic=periodic)
    s.add_species("Ar", Q(39.948, "u"))
    s.add_species("Kr", Q(83.8, "u"))
    s.add("Ar", rng.uniform(0, 4e-9, (300, 3)))
    s.add("Kr", rng.uniform(0, 4e-9, (200, 3)))
    params = {"Ar": (1.65e-21, 3.4e-10), "Kr": (2.3e-21, 3.6e-10)}
    F1, U1 = LennardJones(params, pairlist_threshold=10 ** 9).compute(s)
    F2, U2 = LennardJones(params, pairlist_threshold=2).compute(s)
    assert U2 == pytest.approx(U1, rel=1e-10)
    assert np.allclose(F2, F1, rtol=1e-8, atol=1e-12 * np.max(np.abs(F1)))
