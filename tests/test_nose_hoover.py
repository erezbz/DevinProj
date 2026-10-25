import numpy as np

from accusim import LennardJones, NoseHoover, Q, Simulation, System, k_B


def _argon(n_cells=3, T0=60.0):
    m = Q(39.948, "u").value
    L = (4 * n_cells ** 3 * m / 1374.0) ** (1 / 3)
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) + 0.25
    pts = np.array([b + [i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells) for b in base])
    s = System(3, box=L, periodic=True)
    s.add_species("Ar", m)
    s.add("Ar", pts * L / n_cells)
    s.maxwell_boltzmann(T0, np.random.default_rng(3))
    return s, LennardJones({"Ar": (120 * k_B.value, 3.4e-10)}, cutoff=2.5)


def _run(dt, steps):
    s, lj = _argon()
    nh = NoseHoover(120, Q(0.2, "ps"))
    sim = Simulation(s, [lj], integrator=nh, dt=Q(dt, "fs"))
    sim.run(steps=steps, record_every=10, positions=False)
    return nh, {c.name: c for c in sim.accuracy_report().checks}


def test_nose_hoover_reaches_target_and_conserves_extended_energy():
    nh, checks = _run(5, 4000)
    assert checks["energy"].value is None  # not conserved by design
    assert checks["thermostat"].value < 0.05  # heated 60 K -> 120 K
    assert checks["extended energy"].value < 1e-3
    assert nh.eta != 0.0
    # the extended-energy error is integration error: it shrinks ~dt^2 (observed 3.4x per halving)
    _, fine = _run(2.5, 8000)
    assert checks["extended energy"].value / fine["extended energy"].value > 2.5
