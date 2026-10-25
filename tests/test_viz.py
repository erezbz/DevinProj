import numpy as np

from accusim import Q, Simulation, System
from accusim.viz import animate, animate3d, plot_energy


def test_animations_and_plots_render(tmp_path):
    s = System(3, box=Q(10, "nm"), periodic=True)
    s.add_species("A", Q(10, "u"))
    s.add("A", np.random.default_rng(0).uniform(0, 1e-8, (20, 3)))
    s.maxwell_boltzmann(Q(300, "K"))
    sim = Simulation(s, dt=1e-13)
    tr = sim.run(steps=10)
    animate(tr, s, str(tmp_path / "a.gif"), unit=1e-9)
    animate3d(tr, s, str(tmp_path / "b.gif"), unit=1e-9)
    plot_energy(tr, str(tmp_path / "e.png"))
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.gif", "b.gif", "e.png"))
