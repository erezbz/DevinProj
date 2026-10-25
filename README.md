# accusim

**An accuracy-first universal particle simulator: it tells you how wrong your simulation is.**

You describe particles of any kind (atoms, molecules, cells, planets), the forces between them, and the reactions they
can undergo, all in one model with physical units. accusim runs it, and then gives you an **accuracy report**:
which conservation laws *should* hold for the physics you configured, how far each one drifted,
whether Newtonian mechanics is still valid at the speeds reached, and an estimate of the timestep error.
That estimate comes from re-running at dt/2 and dt/4 and measuring the integrator's observed order of convergence.

```python
from accusim import *

s = System(3)
s.add_species("Sun", Q(1, "Msun"))
s.add_species("Earth", Q(1, "Mearth"))
s.add("Sun", [0, 0, 0])
s.add("Earth", Q([1, 0, 0], "au"), Q([0, 29.78, 0], "km/s"))

sim = Simulation(s, [Gravity()], integrator=Yoshida4())
sim.dt = suggest_dt(sim)                  # from the physical timescales present
sim.run(duration=Q(10, "yr"))
print(sim.accuracy_report(convergence_duration=Q(1, "yr")))
```

```
Accuracy report (all applicable checks passed)
  [PASS] units                              every input was dimension-checked on entry
  [PASS] energy                   1.04e-11  max |E(t)-E0| / energy scale (incl. reaction ledger)
  [PASS] momentum                 1.85e-14  max |P(t)-P0| / sum m|v|
  [PASS] angular momentum         6.80e-14  max |L(t)-L0| / sum |r x p|
  [PASS] mass-energy              0.00e+00  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [PASS] relativity               3.78e-08  (v/c)^2 of fastest particle: Newtonian error ~ this size
  [PASS] timestep error           1.29e-08  est. position error at dt, observed order 4.0/4
```

## What's inside

| Area | Features |
|---|---|
| Units | Every input is a `Quantity` with a dimension. `Q(1, "m") + Q(1, "s")` raises `UnitError`. CODATA constants. |
| Forces | Newtonian gravity (softening optional), Coulomb, Lennard-Jones (mixing rules, shifted cutoff), harmonic bonds, uniform g/E fields, drag. Periodic boxes with minimum image, or reflecting walls. |
| Integrators | Velocity Verlet (2nd order, symplectic), Yoshida (4th order, symplectic), Langevin BAOAB (thermostat), Brownian/overdamped (diffusing molecules). |
| Reactions | `A -> B`, `A -> B + C`, `A + B -> C`, `A + B -> C + D`, with rates and reaction radii (Doi model). Exact conservation of momentum and charge. Mass-energy is conserved: Q = Δm·c² + chemical energy is shared in the center-of-mass frame, energy thresholds are respected, and products store internal energy that later reactions can use. |
| Kinetics | Exact Gillespie SSA for well-mixed systems, to cross-check particle simulations. |
| Accuracy | Applicability-aware conservation checks, relativistic-speed check, reaction-timestep bias check, `calibrate_dt` (halves dt until energy tolerance met), Richardson timestep-error estimate with observed order, pre-flight `suggest_dt`. |
| Viz | GIF animations and energy-error plots (matplotlib). |

## No-code models: JSON + command line

Describe a whole simulation in JSON, with units written as strings, then run it:

```
python -m accusim examples/models/ozone.json [--convergence] [--gif out.gif]
```

```
450 particles, 3 species, dt = 5e-11 s
final counts: {'O': 112, 'O2': 262, 'O3': 38}
Accuracy report (all applicable checks passed)
  [PASS] energy                   2.93e-15  ...
```

## Performance

Lennard-Jones automatically switches to an O(N) cell list once there are 400 or more particles.
One force evaluation takes 0.044 s for 2,000 atoms (0.36 s with all pairs) and 0.17 s for 8,000 atoms.

## Examples (`examples/`)

- `solar_system.py`: Sun, the inner planets and Jupiter for 20 years. Earth's simulated year is 365.27 days, energy drift is 1e-11, and observed order is 4.0 (expected 4).
- `argon_gas.py`: 216 argon atoms with real Lennard-Jones parameters. Energy is conserved to 2e-5 at dt = 3.2 fs.
- `chemistry.py`: reversible A + B ⇌ C with 0.5 eV of binding energy. The particle simulation gives C = 114 and exact Gillespie kinetics gives C = 111.
- `enzyme.py`: **biology**. Michaelis-Menten E + S ⇌ ES → E + P with diffusing molecules (Brownian dynamics).
  The particle simulation gives P = 242, against 237 ± 9 from exact Gillespie.
- `star_cluster.py`: **astronomy**. A 400-star Plummer cluster run for 2 crossing times. The virial ratio goes from 1.04 to 1.02,
  energy drift is 1e-7, and angular momentum holds to 3e-16. Add `--gif` for a rotating 3D animation.
- `fusion.py`: D + T → He-4 + n. The 17.589 MeV comes out of the mass defect automatically, and the neutron gets 14.05 MeV. The report also flags that the neutron's 0.17 c makes Newtonian kinematics about 3% off.

Add `--gif` to write an animation to `examples/out/`.

## Tests

`python -m pytest`. The 48 tests cover:
- **Forces:** F = −∇U is checked numerically for every force (open and periodic), plus Newton's third law.
- **Dynamics against exact answers:** Kepler orbits, energy bounded over 100 eccentric orbits, the measured integrator order, and the exact harmonic-oscillator solution.
- **Engine and models:** the cell list matches the all-pairs results, and JSON models run end to end.
- **Reactions and kinetics:** exponential decay, conservation during reactions, fission energy sharing, the D-T Q-value, endothermic thresholds, Gillespie against the rate equation, and the Brownian diffusion coefficient.

## Limits (honest)

- Gravity and Coulomb are still O(N²), so they are practical up to a few thousand bodies. A Barnes-Hut tree is the next step. Lennard-Jones is O(N).
- Kinematics are Newtonian. The relativity check tells you when that stops being valid.
- Reactions use the Doi contact model; diffusion-limited regimes need a small dt relative to the time particles take to cross the reaction radius.
