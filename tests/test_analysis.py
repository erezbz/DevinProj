import numpy as np

from accusim import Q, Simulation, System


def test_wall_pressure_matches_kinetic_theory():
    from accusim.analysis import wall_pressure
    s = System(3, box=[1e-8, 1.5e-8, 2e-8], periodic=False)
    s.add_species("Ar", Q(39.948, "u"))
    rng = np.random.default_rng(4)
    s.add("Ar", rng.uniform(0, 1, (2000, 3)) * s.box)
    s.maxwell_boltzmann(300, rng=rng)
    sim = Simulation(s, [], dt=Q(2, "fs"))
    s.wall_impulse = 0.0
    sim.run(steps=100000 // 2, record_every=10000)
    expected = 2 * s.kinetic_energy() / (3 * np.prod(s.box))
    assert abs(wall_pressure(s, 1e-10) / expected - 1) < 0.04
