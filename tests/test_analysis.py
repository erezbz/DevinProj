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


def test_green_kubo_matches_langevin_diffusion():
    from accusim.analysis import green_kubo_diffusion, vacf
    from accusim.integrators import Langevin
    s = System(3)
    s.add_species("Ar", Q(39.948, "u"))
    s.add("Ar", np.zeros((400, 3)))
    rng = np.random.default_rng(8)
    s.maxwell_boltzmann(300, rng=rng)
    gamma = 1e12
    sim = Simulation(s, [], integrator=Langevin(300, gamma), dt=Q(20, "fs"), seed=8)
    tr = sim.run(steps=4000, record_every=5)
    t, C = vacf(tr, max_lag=120)
    m = s.species[0].mass
    kT = 1.380649e-23 * 300
    assert abs(C[0] / (3 * kT / m) - 1) < 0.05
    assert abs(green_kubo_diffusion(tr, max_lag=120) / (kT / (m * gamma)) - 1) < 0.06
