import numpy as np

from accusim import Q, Reaction, Simulation, System
from accusim.io import load, save


def test_roundtrip_with_changing_particle_count(tmp_path):
    s = System(3, box=1e-8, periodic=True)
    s.add_species("A", Q(10, "u"))
    s.add_species("B", Q(10, "u"))
    s.add("A", np.random.default_rng(0).uniform(0, 1e-8, (30, 3)))
    sim = Simulation(s, reactions=[Reaction("A", "B", rate=1e12)], dt=1e-13, seed=0)
    tr = sim.run(steps=10)
    save(tr, s, tmp_path / "run.npz")
    tr2, meta = load(tmp_path / "run.npz")
    assert meta[0]["name"] == "A" and tr2.counts == tr.counts
    assert all(np.array_equal(a, b) for a, b in zip(tr.positions, tr2.positions))
    assert np.allclose(tr2.total_energy, tr.total_energy)
