import numpy as np

from accusim import LennardJones, Rattle, Simulation, System


def test_rigid_rotor_keeps_length_and_energy():
    s = System(3)
    s.add_species("A", 1.0)
    s.add_species("B", 3.0)
    a = s.add("A", [[0.0, 0.0, 0.0]], vel=[[0.0, 3.0, 0.0]])[0]
    b = s.add("B", [[1.0, 0.0, 0.0]], vel=[[0.0, -1.0, 0.0]])[0]
    sim = Simulation(s, [], integrator=Rattle([(a, b, 1.0)]), dt=0.01)
    sim.run(steps=2000, positions=False)
    checks = {c.name: c for c in sim.accuracy_report().checks}
    assert checks["constraints"].value < 1e-10
    assert checks["energy"].value < 1e-9
    assert checks["angular momentum"].value < 1e-9


def _collision(dt):
    s = System(3)
    s.add_species("A", 1.0)
    s.add_species("C", 1.0)
    a, b = s.add("A", [[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])
    s.add("C", [[3.0, 0.3, 0.0]], vel=[[-1.0, 0.0, 0.0]])
    lj = LennardJones({"A": (1.0, 1.0), "C": (1.0, 1.0)}, cutoff=2.5)
    sim = Simulation(s, [lj], integrator=Rattle([(a, b, 1.0)]), dt=dt)
    sim.run(steps=int(round(6 / dt)), positions=False)
    return s, {c.name: c for c in sim.accuracy_report().checks}


def test_rigid_molecule_collision_conserves_energy_and_spins_up():
    s, checks = _collision(1e-3)
    assert checks["constraints"].value < 1e-10
    assert checks["energy"].value < 1e-4
    assert checks["momentum"].value < 1e-12
    w = np.cross(s.pos[0] - s.pos[1], s.vel[0] - s.vel[1])
    assert np.linalg.norm(w) > 1e-3  # the off-centre hit set the dumbbell spinning
    # constraint forces do no work: the remaining energy error is 2nd-order integration error (observed 3.5x/halving)
    _, fine = _collision(5e-4)
    assert checks["energy"].value / fine["energy"].value > 3


def test_matrix_and_gauss_seidel_shake_agree_on_rigid_water():
    import os
    import sys

    from accusim import Q
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from rigid_water import build
    out = []
    for method in ("gauss-seidel", "matrix"):
        s, forces, cons = build()
        sim = Simulation(s, forces, integrator=Rattle(cons, method=method), dt=Q(2, "fs"))
        sim.run(steps=50, positions=False)
        out.append((s.pos.copy(), sim.integrator.max_violation))
    assert np.max(np.abs(out[0][0] - out[1][0])) < 1e-8 * 1e-10
    assert max(out[0][1], out[1][1]) < 1e-11
