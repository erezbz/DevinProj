import json
import os

from accusim.__main__ import main
from accusim.model import build

HERE = os.path.dirname(__file__)


def test_json_model_runs_and_conserves(tmp_path):
    sim = main([os.path.join(HERE, "..", "examples", "models", "ozone.json")])
    rep = sim.accuracy_report()
    assert rep.ok, str(rep)
    assert sim.reactions.events["O + O2 -> O3"] > 0


def test_model_with_forces_and_kj_per_mol():
    model = {
        "box": "3 nm", "periodic": True, "seed": 1,
        "species": {"Ar": {"mass": "39.948 u"}},
        "particles": [{"species": "Ar", "count": 50, "random": True}],
        "temperature": "150 K",
        "forces": [{"type": "LennardJones", "params": {"Ar": ["0.996 kJ/mol", "3.405 A"]}}],
        "dt": "2 fs",
    }
    sim, _ = build(json.loads(json.dumps(model)))
    assert abs(sim.forces[0].params["Ar"][0] - 1.654e-21) < 1e-24
    sim.run(steps=5, positions=False)


def test_json_molecules_build_bonds_angles_and_exclusions():
    import os

    import numpy as np

    from accusim import Coulomb, HarmonicAngle, HarmonicBond
    from accusim.model import load
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "models", "water.json")
    sim, model = load(path)
    s = sim.system
    assert s.count("O") == 27 and s.count("H") == 54
    bonds = [f for f in sim.forces if isinstance(f, HarmonicBond)]
    angles = [f for f in sim.forces if isinstance(f, HarmonicAngle)]
    coul = [f for f in sim.forces if isinstance(f, Coulomb)][0]
    assert sum(len(b.bonds) for b in bonds) == 54 and sum(len(a.angles) for a in angles) == 27
    assert len(coul.exclude) == 27 * 3
    w = {pid: n for n, pid in enumerate(s.ids)}
    r = [np.linalg.norm(s.pos[w[j]] - s.pos[w[i]]) for i, j in bonds[0].bonds]
    assert np.allclose(r, 1.012e-10, rtol=1e-3)  # templates placed rigidly at rest geometry
    assert abs(angles[0].k - 317.6e3 / 6.02214076e23) < 1e-24
    sim.run(steps=200, positions=False)
    rep = sim.accuracy_report()
    assert rep["momentum"].status == "PASS" and rep["angular momentum"].status == "PASS"
    assert rep["energy"].value < 1e-3


def test_json_nose_hoover_integrator():
    model = {
        "box": "3 nm", "periodic": True, "seed": 1,
        "species": {"Ar": {"mass": "39.948 u"}},
        "particles": [{"species": "Ar", "count": 50, "random": True}],
        "temperature": "150 K",
        "forces": [{"type": "LennardJones", "params": {"Ar": ["0.996 kJ/mol", "3.405 A"]}}],
        "integrator": {"type": "NoseHoover", "temperature": "150 K", "period": "0.5 ps"},
        "dt": "2 fs",
    }
    sim, _ = build(json.loads(json.dumps(model)))
    assert abs(sim.integrator.tau - 0.5e-12) < 1e-18
    sim.run(steps=20, positions=False)
    assert "extended energy" in {c.name for c in sim.accuracy_report().checks}
