"""Declarative models: describe a whole simulation in JSON (units as strings).

{
  "dim": 3, "box": "20 nm", "periodic": true,
  "species": {"A": {"mass": "20 u"}, "C": {"mass": "50 u"}},
  "particles": [{"species": "A", "count": 100, "random": true}, {"species": "C", "pos": ["0 nm", "0 nm", "0 nm"]}],
  "temperature": "300 K",
  "forces": [{"type": "LennardJones", "params": {"A": ["1 kJ/mol", "3.4 A"]}}],
  "reactions": [{"reactants": ["A", "B"], "products": ["C"], "rate": "1e8 1/s", "radius": "4 nm"}],
  "integrator": {"type": "VelocityVerlet"},
  "dt": "10 fs", "duration": "1 ps", "seed": 1
}
"""
from __future__ import annotations

import json

import numpy as np

from . import forces as F
from . import integrators as I
from .reactions import Reaction
from .simulation import Simulation
from .system import System
from .units import N_A, Q, Quantity, parse_unit


def qty(x):
    """'3 km/s' -> Quantity; numbers pass through as SI; lists map element-wise."""
    if isinstance(x, str):
        num, _, unit = x.strip().partition(" ")
        return Q(float(num), unit.strip() or "1")
    if isinstance(x, list):
        items = [qty(v) for v in x]
        if all(isinstance(v, Quantity) for v in items):
            dims = {v.dim for v in items}
            if len(dims) != 1:
                raise ValueError(f"mixed units in {x}")
            return Quantity(np.array([v.value for v in items]), items[0].dim)
        return np.array(items, dtype=float)
    return x


def _energy(x):
    q = qty(x)
    if isinstance(q, Quantity) and q.dim == parse_unit("J/mol").dim:
        return q / N_A
    return q


from .tree import BarnesHut  # noqa: E402

FORCE_TYPES = {"BarnesHut": BarnesHut, "Gravity": F.Gravity, "Coulomb": F.Coulomb, "Ewald": F.Ewald, "LennardJones": F.LennardJones,
               "HarmonicBond": F.HarmonicBond, "UniformField": F.UniformField, "Drag": F.Drag}
INTEGRATORS = {"VelocityVerlet": I.VelocityVerlet, "Yoshida4": I.Yoshida4, "Langevin": I.Langevin,
               "Brownian": I.Brownian}


def build(model: dict) -> tuple[Simulation, dict]:
    s = System(model.get("dim", 3), box=qty(model["box"]) if "box" in model else None,
               periodic=model.get("periodic", False))
    for name, sp in model["species"].items():
        s.add_species(name, **{k: qty(v) for k, v in sp.items()})
    rng = np.random.default_rng(model.get("seed"))
    for p in model.get("particles", []):
        if p.get("random"):
            if s.box is None:
                raise ValueError("random placement needs a box")
            s.add(p["species"], rng.uniform(0, 1, (p["count"], s.dim)) * s.box)
        else:
            s.add(p["species"], qty(p["pos"]), qty(p["vel"]) if "vel" in p else None)
    if "temperature" in model:
        s.maxwell_boltzmann(qty(model["temperature"]), rng)
    forces = []
    for f in model.get("forces", []):
        f = dict(f)
        kind = FORCE_TYPES[f.pop("type")]
        if kind is F.LennardJones:
            f["params"] = {k: (_energy(e), qty(sg)) for k, (e, sg) in f["params"].items()}
        else:
            f = {k: qty(v) for k, v in f.items()}
        forces.append(kind(**f))
    reactions = [Reaction(**{k: (qty(v) if k in ("rate", "radius", "separation") else
                                 _energy(v) if k == "energy" else v) for k, v in r.items()})
                 for r in model.get("reactions", [])]
    integ = dict(model.get("integrator", {"type": "VelocityVerlet"}))
    integrator = INTEGRATORS[integ.pop("type")](**{k: qty(v) for k, v in integ.items()})
    sim = Simulation(s, forces, reactions, integrator, dt=qty(model["dt"]) if "dt" in model else None,
                     seed=model.get("seed"))
    if sim.dt is None:
        from .advisor import suggest_dt
        sim.dt = suggest_dt(sim)
    return sim, model


def load(path):
    with open(path) as fh:
        return build(json.load(fh))
