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


def test_dsf_from_json_model(tmp_path):
    import json

    from accusim.model import load
    n = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
    odd = n.sum(1) % 2 == 1
    g = n * 2.82e-10
    model = {
        "dim": 3, "box": "11.28 A", "periodic": True, "seed": 1, "temperature": "1100 K",
        "species": {"Na": {"mass": "22.99 u", "charge": "1 e"}, "Cl": {"mass": "35.45 u", "charge": "-1 e"}},
        "particles": [{"species": "Na", "pos": g[~odd].tolist()}, {"species": "Cl", "pos": g[odd].tolist()}],
        "forces": [{"type": "LennardJones", "params": {"Na": ["1.4754 kJ/mol", "2.159 A"], "Cl": ["0.0128 kJ/mol", "4.830 A"]},
                    "cutoff": 2.5},
                   {"type": "DampedShiftedCoulomb", "rcut": "5.5 A"}],
        "dt": "1 fs", "duration": "20 fs",
    }
    path = tmp_path / "salt.json"
    path.write_text(json.dumps(model))
    sim, _ = load(str(path))
    tr = sim.run(steps=20)
    E = np.asarray(tr.total_energy)
    assert np.all(np.isfinite(E)) and abs(E[-1] / E[0] - 1) < 1e-3
    model["molecules"] = [{"atoms": [["Na", [0, 0, 0]], ["Cl", [2.8e-10, 0, 0]]], "bonds": [[0, 1, "100 N/m", "2.8 A"]],
                           "at": [[5e-10, 5e-10, 5e-10]]}]
    path.write_text(json.dumps(model))
    sim, _ = load(str(path))
    dsf = [f for f in sim.forces if isinstance(f, DampedShiftedCoulomb)][0]
    assert len(dsf.exclude) == 1


def test_dsf_exclusions_same_in_dense_and_cell_paths():
    s = rocksalt(8)
    rng = np.random.default_rng(5)
    s.pos = (s.pos + rng.normal(0, 0.2, s.pos.shape)) % s.box
    ex = [(int(s.ids[2 * m]), int(s.ids[2 * m + 1])) for m in range(50)]
    fd, ud = DampedShiftedCoulomb(rcut=2.5, k=1.0, pairlist_threshold=10 ** 9, exclude=ex).compute(s)
    fc, uc = DampedShiftedCoulomb(rcut=2.5, k=1.0, pairlist_threshold=1, exclude=ex).compute(s)
    _, u0 = DampedShiftedCoulomb(rcut=2.5, k=1.0).compute(s)
    assert abs(uc / ud - 1) < 1e-12 and np.allclose(fc, fd, rtol=1e-9, atol=1e-12) and abs(u0 - ud) > 1e-6
