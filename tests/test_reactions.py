import numpy as np
import pytest

from accusim import Q, Reaction, Simulation, System, gillespie
from accusim.integrators import Brownian


def gas(n_a=200, n_b=200, box=1e-7, seed=0):
    s = System(3, box=box, periodic=True)
    s.add_species("A", Q(10, "u"))
    s.add_species("B", Q(20, "u"))
    s.add_species("C", Q(30, "u"))
    rng = np.random.default_rng(seed)
    s.add("A", rng.uniform(0, box, (n_a, 3)))
    s.add("B", rng.uniform(0, box, (n_b, 3)))
    s.maxwell_boltzmann(Q(300, "K"), rng)
    return s


def test_decay_follows_exponential_law():
    s = System(3, box=1e-7, periodic=True)
    s.add_species("A", Q(10, "u"))
    s.add_species("A*", Q(10, "u"))
    s.add("A", np.random.default_rng(0).uniform(0, 1e-7, (2000, 3)))
    k = 1e9
    sim = Simulation(s, reactions=[Reaction("A", "A*", rate=k)], dt=1e-11, seed=1)
    sim.run(steps=100, positions=False)
    expected = 2000 * np.exp(-k * 1e-9)
    assert s.count("A") == pytest.approx(expected, abs=4 * np.sqrt(expected))


def test_mass_increase_without_energy_is_forbidden():
    s = gas(n_a=50, n_b=0)
    sim = Simulation(s, reactions=[Reaction("A", "B", rate=1e30)], dt=1e-12, seed=1)
    sim.run(steps=3, positions=False)
    assert s.count("A") == 50


def test_merge_conserves_momentum_mass_charge_and_energy_books():
    s = gas()
    sim = Simulation(s, reactions=[Reaction(["A", "B"], ["C"], rate=1e12, radius=Q(3, "nm"), energy=Q(1, "eV"))],
                     dt=1e-12, seed=2)
    sim.run(steps=200, positions=False)
    assert sim.reactions.events["A + B -> C"] > 20
    rep = sim.accuracy_report()
    for name in ("energy", "momentum", "mass-energy", "charge"):
        assert rep[name].status == "PASS", str(rep)


def test_fission_shares_energy_in_center_of_mass_frame():
    s = System(3)
    s.add_species("P", Q(30, "u"))
    s.add_species("D1", Q(10, "u"))
    s.add_species("D2", Q(20, "u"))
    s.add("P", [0, 0, 0], [100.0, 0, 0])
    sim = Simulation(s, reactions=[Reaction("P", ["D1", "D2"], rate=1e30, energy=Q(2, "eV"))], dt=1e-15, seed=0)
    sim.run(steps=1, positions=False)
    assert s.counts() == {"P": 0, "D1": 1, "D2": 1}
    P = s.momentum()
    assert np.allclose(P, [30 * 1.66053906660e-27 * 100, 0, 0], rtol=1e-10)
    v_cm = P / s.masses.sum()
    ke_rel = 0.5 * np.sum(s.masses[:, None] * (s.vel - v_cm) ** 2)
    assert ke_rel == pytest.approx(Q(2, "eV").value, rel=1e-10)


def test_deuterium_tritium_fusion_releases_17_6_MeV_from_mass_defect():
    s = System(3)
    s.add_species("D", Q(2.01410177812, "u"), charge=Q(1, "e"))
    s.add_species("T", Q(3.0160492779, "u"), charge=Q(1, "e"))
    s.add_species("He4", Q(4.00260325413, "u"), charge=Q(2, "e"))
    s.add_species("n", Q(1.00866491595, "u"))
    s.add("D", [0, 0, 0])
    s.add("T", [1e-15, 0, 0])
    r = Reaction(["D", "T"], ["He4", "n"], rate=1e30, radius=Q(10, "fm"))
    sim = Simulation(s, reactions=[r], dt=1e-25, seed=0)
    sim.run(steps=1, positions=False)
    assert r.q_value / Q(1, "MeV").value == pytest.approx(17.589, abs=1e-3)
    ke = {sp.name: 0.5 * sp.mass * np.sum(v ** 2) for sp, v in zip([s.species[i] for i in s.sp], s.vel)}
    # Non-relativistic two-body split for reactants at rest: Q * m_He / (m_He + m_n).
    assert ke["n"] / Q(1, "MeV").value == pytest.approx(14.049, abs=0.002)
    assert sim.accuracy_report()["mass-energy"].status == "PASS"


def test_endothermic_reaction_respects_threshold():
    s = System(3)
    s.add_species("A", Q(10, "u"))
    s.add_species("B", Q(10, "u"))
    s.add("A", [0, 0, 0], [10.0, 0, 0])
    sim = Simulation(s, reactions=[Reaction("A", "B", rate=1e30, energy=Q(-1, "eV"))], dt=1e-15, seed=0)
    sim.run(steps=5, positions=False)
    assert s.counts() == {"A": 1, "B": 0}


def test_charge_violation_is_rejected():
    s = System(3)
    s.add_species("A", Q(1, "u"), charge=Q(1, "e"))
    s.add_species("B", Q(1, "u"))
    with pytest.raises(ValueError, match="charge"):
        Simulation(s, reactions=[Reaction("A", "B", rate=1)])


def test_gillespie_matches_mass_action_ode():
    k, V, n0 = 1e-20, 1e-18, 1000
    rng = np.random.default_rng(3)
    finals = [gillespie({"A": n0, "B": n0}, [(("A", "B"), ("C",), k)], V, t_end=0.5, rng=rng)[1]["A"][-1]
              for _ in range(20)]
    a_exact = n0 / (1 + k / V * n0 * 0.5)
    assert np.mean(finals) == pytest.approx(a_exact, rel=0.03)


def test_brownian_diffusion_coefficient():
    s = System(3)
    D = 1e-9
    s.add_species("X", Q(100, "u"), diffusion=Q(D, "m^2/s"))
    s.add("X", np.zeros((4000, 3)))
    sim = Simulation(s, integrator=Brownian(Q(300, "K")), dt=1e-9, seed=4)
    sim.run(steps=100, positions=False)
    msd = np.mean(np.sum(s.pos ** 2, axis=1))
    assert msd == pytest.approx(6 * D * 1e-7, rel=0.05)


def test_reversible_reaction_stores_and_returns_internal_energy():
    s = System(3)
    s.add_species("A", Q(10, "u"))
    s.add_species("B", Q(10, "u"))
    s.add_species("C", Q(20, "u"))
    s.add("A", [0, 0, 0])
    s.add("B", [1e-10, 0, 0])
    rx = [Reaction(["A", "B"], "C", rate=1e30, radius=Q(1, "nm"), energy=Q(1, "eV")),
          Reaction("C", ["A", "B"], rate=1e30, energy=Q(-1, "eV"))]
    sim = Simulation(s, reactions=rx, dt=1e-15, seed=0)
    sim.run(steps=1, positions=False)
    assert s.counts()["C"] == 1 and s.eint[0] == pytest.approx(Q(1, "eV").value)
    sim.run(steps=1, positions=False)
    assert s.counts()["C"] == 0
    assert s.kinetic_energy() == pytest.approx(0, abs=1e-30)
    assert sim.accuracy_report()["energy"].status == "PASS"


def test_report_warns_about_relativistic_speeds():
    s = System(3)
    s.add_species("n", Q(1, "u"))
    s.add("n", [0, 0, 0], [0.2 * 2.998e8, 0, 0])
    sim = Simulation(s, dt=1e-20)
    sim.run(steps=2, positions=False)
    assert sim.accuracy_report()["relativity"].status == "FAIL"


def test_report_flags_coarse_reaction_timestep():
    s = gas(n_a=20, n_b=20)
    rx = [Reaction(["A", "B"], ["C"], rate=1e12, radius=Q(1, "nm"))]
    sim = Simulation(s, reactions=rx, dt=1e-12, seed=0)
    sim.run(steps=2, positions=False)
    assert sim.accuracy_report()["reaction timestep"].status == "FAIL"
