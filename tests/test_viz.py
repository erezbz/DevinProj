import numpy as np

from accusim import Q, Simulation, System
from accusim.viz import animate, animate3d, export_html, plot_energy


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
    export_html(tr, s, str(tmp_path / "v.html"))
    html = (tmp_path / "v.html").read_text()
    assert "requestAnimationFrame" in html and '"species":["A"]' in html and "%" not in html.split("<script>")[0][-5:]
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.gif", "b.gif", "e.png", "v.html"))


def test_html_viewer_includes_bonds_and_energy_panel(tmp_path):
    import json
    import os
    import re

    from accusim.model import load
    sim, _ = load(os.path.join(os.path.dirname(__file__), "..", "examples", "models", "water.json"))
    tr = sim.run(steps=20)
    out = tmp_path / "w.html"
    export_html(tr, sim.system, str(out), sim=sim)
    data = json.loads(re.search(r"const D=(\{.*?\});const C=", out.read_text()).group(1))
    assert len(data["bonds"]) == 54 and len(data["energy"]) == 21 and data["emax"] > 0
