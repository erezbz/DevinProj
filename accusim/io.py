"""Save/load trajectories (compressed .npz) so long runs can be analysed or replayed later."""
from __future__ import annotations

import json

import numpy as np

from .simulation import Trajectory

_SCALARS = ("t", "kinetic", "potential", "internal", "released", "mass", "charge")


def save(trajectory: Trajectory, system, path):
    data = {k: np.asarray(getattr(trajectory, k)) for k in _SCALARS}
    data["momentum"] = np.asarray(trajectory.momentum)
    data["angular_momentum"] = np.asarray(trajectory.angular_momentum)
    data["counts"] = np.frombuffer(json.dumps(trajectory.counts).encode(), dtype=np.uint8)
    data["species_meta"] = np.frombuffer(json.dumps([sp.__dict__ for sp in system.species]).encode(), np.uint8)
    if trajectory.positions:
        lens = np.array([len(p) for p in trajectory.positions])
        data["frame_len"] = lens
        data["positions"] = np.concatenate(trajectory.positions)
        data["species"] = np.concatenate(trajectory.species)
    np.savez_compressed(path, **data)


def load(path) -> tuple[Trajectory, list]:
    z = np.load(path)
    tr = Trajectory()
    for k in _SCALARS:
        setattr(tr, k, z[k].tolist())
    tr.momentum = list(z["momentum"])
    tr.angular_momentum = list(z["angular_momentum"])
    tr.counts = json.loads(bytes(z["counts"]).decode())
    if "frame_len" in z:
        cuts = np.cumsum(z["frame_len"])[:-1]
        tr.positions = np.split(z["positions"], cuts)
        tr.species = np.split(z["species"], cuts)
    return tr, json.loads(bytes(z["species_meta"]).decode())
