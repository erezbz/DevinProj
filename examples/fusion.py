"""D + T -> He-4 + n: the 17.6 MeV comes out of the mass defect automatically."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from accusim import Q, Reaction, Simulation, System  # noqa: E402

s = System(3)
s.add_species("D", Q(2.01410177812, "u"), charge=Q(1, "e"))
s.add_species("T", Q(3.0160492779, "u"), charge=Q(1, "e"))
s.add_species("He4", Q(4.00260325413, "u"), charge=Q(2, "e"))
s.add_species("n", Q(1.00866491595, "u"))
s.add("D", [0, 0, 0])
s.add("T", [2e-15, 0, 0])
r = Reaction(["D", "T"], ["He4", "n"], rate=1e30, radius=Q(10, "fm"))
sim = Simulation(s, reactions=[r], dt=1e-24, seed=0)
sim.run(steps=1)
print(f"Q-value from masses: {r.q_value / Q(1, 'MeV').value:.3f} MeV (textbook: 17.6 MeV)")
for i, v in zip(s.sp, s.vel):
    sp = s.species[i]
    print(f"  {sp.name:>4}: {0.5 * sp.mass * v @ v / Q(1, 'MeV').value:.3f} MeV, speed {np.linalg.norm(v) / 2.998e8:.3f} c")
print(sim.accuracy_report())
