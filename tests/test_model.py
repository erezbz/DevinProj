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
