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
        if not x:
            return np.zeros(0)
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


from .tree import BarnesHut

FORCE_TYPES = {"BarnesHut": BarnesHut, "Gravity": F.Gravity, "Coulomb": F.Coulomb, "Ewald": F.Ewald, "DampedShiftedCoulomb": F.DampedShiftedCoulomb, "LennardJones": F.LennardJones,
               "HarmonicBond": F.HarmonicBond, "HarmonicAngle": F.HarmonicAngle, "UniformField": F.UniformField, "Drag": F.Drag}
INTEGRATORS = {"VelocityVerlet": I.VelocityVerlet, "Yoshida4": I.Yoshida4, "Langevin": I.Langevin,
               "Brownian": I.Brownian}


def _random_rotation(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q * np.sign(np.diag(r))


def _molecules(s, specs, rng):
    """Place copies of molecule templates; return their bond/angle forces and intramolecular exclusions.

    Template: {"atoms": [[species, [x, y, z]], ...] (positions with units, relative),
               "bonds": [[i, j, k, r0], ...], "angles": [[i, j, l, k, theta0_degrees], ...],
               "count": n, "random": true}  or  "at": [[x, y, z], ...] instead of count/random.
    Atoms in the same molecule never interact through Coulomb (standard force-field exclusion).
    """
    forces, exclude = [], []
    for m in specs:
        species = [a[0] for a in m["atoms"]]
        rel = np.array([np.asarray(_si(qty(a[1])), dtype=float) for a in m["atoms"]])
        rel -= rel.mean(axis=0)
        if "at" in m:
            centres = np.array([_si(qty(c)) for c in m["at"]], dtype=float)
        else:
            if s.box is None:
                raise ValueError("random molecule placement needs a box")
            centres = rng.uniform(0, 1, (m["count"], s.dim)) * s.box
        bond_groups, angle_groups = {}, {}
        for c in centres:
            R = _random_rotation(rng, s.dim) if m.get("random", "at" not in m) else np.eye(s.dim)
            ids = [int(s.add(sp, [c + R @ r])[0]) for sp, r in zip(species, rel)]
            for i, j, k, r0 in m.get("bonds", []):
                bond_groups.setdefault((float(_si(qty(k))), float(_si(qty(r0)))), []).append((ids[i], ids[j]))
            for i, j, k3, k, th in m.get("angles", []):
                angle_groups.setdefault((float(_si(_energy(k))), float(np.deg2rad(th))), []).append((ids[i], ids[j], ids[k3]))
            exclude += [(ids[a], ids[b]) for a in range(len(ids)) for b in range(a + 1, len(ids))]
        forces += [F.HarmonicBond(b, k=k, r0=r0) for (k, r0), b in bond_groups.items()]
        forces += [F.HarmonicAngle(a, k=k, theta0=th) for (k, th), a in angle_groups.items()]
    return forces, exclude


def _si(x):
    return x.value if isinstance(x, Quantity) else x


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
    mol_forces, exclude = _molecules(s, model.get("molecules", []), rng)
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
        if kind is F.DampedShiftedCoulomb and exclude:
            raise ValueError("DampedShiftedCoulomb has no intramolecular exclusions yet; use Coulomb with molecules")
        if kind is F.Coulomb and exclude:
            f["exclude"] = list(f.get("exclude", [])) + exclude
        forces.append(kind(**f))
    forces += mol_forces
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
