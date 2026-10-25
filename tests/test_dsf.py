import numpy as np

from accusim import System
from accusim.forces import DampedShiftedCoulomb, Ewald


def rocksalt(n):
    s = System(3, box=float(n), periodic=True)
    s.add_species("Na", 1.0, charge=1.0)
    s.add_species("Cl", 1.0, charge=-1.0)
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    odd = g.sum(1) % 2 == 1
    s.add("Na", g[~odd].astype(float))
    s.add("Cl", g[odd].astype(float))
    return s


def test_dsf_madelung_close_to_ewald():
    s = rocksalt(8)
    _, U = DampedShiftedCoulomb(rcut=3.9, alpha=0.6, k=1.0).compute(s)
    assert abs(-2 * U / s.n / 1.747564594633 - 1) < 2e-3


def test_dsf_force_is_gradient_and_vanishes_at_cutoff():
    s = rocksalt(4)
    rng = np.random.default_rng(1)
    s.pos += rng.normal(0, 0.1, s.pos.shape)
    dsf = DampedShiftedCoulomb(rcut=1.9, k=1.0)
    f, _ = dsf.compute(s)
    h = 1e-6
    for i, ax in [(0, 0), (5, 1), (17, 2)]:
        s.pos[i, ax] += h; _, up = dsf.compute(s)
        s.pos[i, ax] -= 2 * h; _, um = dsf.compute(s)
        s.pos[i, ax] += h
        assert abs(-(up - um) / (2 * h) - f[i, ax]) < 1e-6 * max(1, abs(f[i, ax]))
    assert np.allclose(f.sum(0), 0, atol=1e-10)
    two = System(3)
    two.add_species("A", 1.0, charge=1.0)
    two.add("A", [[0, 0, 0], [1.9 - 1e-9, 0, 0]])
    f2, u2 = dsf.compute(two)
    self_term = 2 * (float(__import__("accusim.forces", fromlist=["_erfc"])._erfc(np.array(dsf.alpha * 1.9))) / 3.8
                     + dsf.alpha / np.sqrt(np.pi))
    assert abs(f2[0, 0]) < 1e-7 and abs(u2 + self_term) < 1e-7


def test_dsf_vs_ewald_on_disordered_melt():
    s = rocksalt(4)
    rng = np.random.default_rng(2)
    s.pos = (s.pos + rng.normal(0, 0.25, s.pos.shape)) % s.box
    _, ue = Ewald(k=1.0).compute(s)
    _, ud = DampedShiftedCoulomb(rcut=1.99, alpha=0.9, k=1.0).compute(s)
    assert abs(ud / ue - 1) < 3e-2


def test_dsf_cell_list_path_matches_dense():
    s = rocksalt(10)
    rng = np.random.default_rng(3)
    s.pos = (s.pos + rng.normal(0, 0.2, s.pos.shape)) % s.box
    fd, ud = DampedShiftedCoulomb(rcut=2.5, k=1.0, pairlist_threshold=10 ** 9).compute(s)
    fc, uc = DampedShiftedCoulomb(rcut=2.5, k=1.0, pairlist_threshold=1).compute(s)
    assert abs(uc / ud - 1) < 1e-12 and np.allclose(fc, fd, rtol=1e-9, atol=1e-12)
