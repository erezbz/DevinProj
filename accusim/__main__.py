"""python -m accusim model.json [--gif out.gif] [--convergence]"""
import argparse

from .model import load, qty


def main(argv=None):
    ap = argparse.ArgumentParser(prog="accusim", description="Run a JSON model and print the accuracy report.")
    ap.add_argument("model")
    ap.add_argument("--gif", help="write an animation")
    ap.add_argument("--html", help="write an interactive 3D viewer (self-contained HTML)")
    ap.add_argument("--3d", dest="three_d", action="store_true", help="3D rotating animation")
    ap.add_argument("--convergence", action="store_true", help="estimate timestep error (3 extra runs)")
    a = ap.parse_args(argv)
    sim, model = load(a.model)
    duration = qty(model["duration"])
    print(f"{sim.system.n} particles, {len(sim.system.species)} species, dt = {sim.dt:.3g} s")
    tr = sim.run(duration=duration, record_every=max(1, int(duration.value / sim.dt) // 200))
    print("final counts:", sim.system.counts())
    conv = duration if a.convergence else None
    print(sim.accuracy_report(convergence_duration=conv))
    if a.gif:
        from .viz import animate, animate3d
        (animate3d if a.three_d else animate)(tr, sim.system, a.gif, title=a.model)
    if a.html:
        from .viz import export_html
        export_html(tr, sim.system, a.html, title=a.model)
    return sim


if __name__ == "__main__":
    main()
