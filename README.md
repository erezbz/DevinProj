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
  [PASS] timestep vs physics      1.62e-02  dt / shortest timescale (closest approach / speed, 8e+05 s) at end of run
  [PASS] relativity               3.78e-08  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 0.000194 c)
  [PASS] timestep error           1.29e-08  est. position error at dt (relative to system size), observed order 4.0/4
```

## What's inside

| Area | Features |
|---|---|
| Units | Every input is a `Quantity` with a dimension. `Q(1, "m") + Q(1, "s")` raises `UnitError`. CODATA constants. |
| Forces | Newtonian gravity (softening optional), Coulomb, Ewald summation for periodic electrostatics (plus `DampedShiftedCoulomb`, a cheaper cutoff alternative validated against it) (reproduces the NaCl Madelung constant to 1e-7), Lennard-Jones (mixing rules, shifted cutoff), harmonic bonds, uniform g/E fields, drag. Periodic boxes with minimum image, or reflecting walls. |
| Integrators | Velocity Verlet (2nd order, symplectic), Yoshida (4th order, symplectic), Langevin BAOAB, Nosé–Hoover (with its conserved extended energy checked in the report) (thermostat), Brownian/overdamped (diffusing molecules). |
| Reactions | `A -> B`, `A -> B + C`, `A + B -> C`, `A + B -> C + D`, with rates and reaction radii (Doi model). Exact conservation of momentum and charge. Mass-energy is conserved: Q = Δm·c² + chemical energy is shared in the center-of-mass frame, energy thresholds are respected, and products store internal energy that later reactions can use. |
| Kinetics | Exact Gillespie SSA for well-mixed systems, to cross-check particle simulations. |
| Accuracy | Applicability-aware conservation checks, dt-vs-physical-timescale stability check, Barnes-Hut force-error check, periodic-Coulomb warning, relativistic-speed check, reaction-timestep bias check, `calibrate_dt` (halves dt until energy tolerance met), Richardson timestep-error estimate with observed order, pre-flight `suggest_dt`. |
| Viz | GIF animations and energy-error plots (matplotlib). |

## It proves itself: `python -m accusim.validate`

```
benchmark                                         expected       accusim    rel.err  result
Circular orbit closes after Kepler period                0   9.06807e-09    9.1e-09  PASS
Harmonic oscillator x(t) = A cos(wt)              0.916093      0.916093    6.3e-13  PASS
Velocity Verlet convergence order                        2       2.00008    4.0e-05  PASS
NaCl Madelung constant (Ewald)                     1.74756       1.74756    5.3e-08  PASS
D-T fusion Q-value from mass defect                17.5893       17.5893    1.8e-07  PASS
First-order decay N(t) = N0 exp(-kt)               7357.59          7278    1.1e-02  PASS
Gillespie A+B->C vs rate equation                  166.667         166.7    2.0e-04  PASS
Einstein diffusion <r^2> = 6Dt                       6e-16   5.99245e-16    1.3e-03  PASS
FIRE: LJ dimer minimum at 2^(1/6) sigma            3.81637       3.81637    2.1e-11  PASS
```

(The decay row is a stochastic sample: 7278 against 7358 is 1.2 standard deviations.)

## No-code models: JSON + command line

Describe a whole simulation in JSON, with units written as strings, then run it:

```
python -m accusim examples/models/ozone.json [--convergence] [--gif out.gif]
```

```
450 particles, 3 species, dt = 1e-11 s
Accuracy report (no failures, 1 warning)
  [PASS] energy                   2.39e-15  ...
  [WARN] reaction timestep        5.00e-02  max rate*dt of contact reactions: rates biased low by ~2%
```

### Molecules in JSON

A `"molecules"` section places copies of a template, either at given centres (`"at"`) or randomly (`"count"`), each with a random orientation.
Bonds, angles and same-molecule Coulomb exclusions are generated automatically. See `examples/models/water.json` (27 flexible SPC/Fw waters):

```json
"molecules": [{
  "atoms":  [["O", ["0 A","0 A","0 A"]], ["H", ["1.012 A","0 A","0 A"]], ["H", ["-0.399 A","0.930 A","0 A"]]],
  "bonds":  [[0, 1, "735.9 N/m", "1.012 A"], [0, 2, "735.9 N/m", "1.012 A"]],
  "angles": [[1, 0, 2, "317.6 kJ/mol", 113.24]],
  "count": 27, "random": true
}]
```

## Preparing and analysing systems

- `minimize(sim)`: FIRE energy minimisation, to relax hand-built structures before dynamics. Tested: it finds the LJ dimer minimum
  at 2^(1/6)σ to 1e-6 and relaxes a distorted triatomic to its rest bond length and angle.
- `rdf(system, rmax)`: radial distribution function g(r). Tested: ideal gas → 1, and the FCC peak at a/√2.
- `msd(trajectory, box)`: mean squared displacement against time, unwrapping periodic boundaries.
- `pressure(sim)`: pressure for *any* force set, from box rescaling (−dU/dV), so no per-force virial code is needed. Tested against NkT/V and an independent pairwise LJ virial (1e-5).
- `accusim.io.save/load`: compressed .npz trajectories, including runs where the particle count changes.

## Interactive 3D viewer

Add `--html out.html` to the CLI, or call `accusim.viz.export_html(trajectory, system, path)`.
It writes one self-contained HTML file with no internet needed: drag to rotate, scroll to zoom, and a timeline slider.
Pass `sim=` (the CLI does this) to draw bonds and a live **energy-error panel**, so the accuracy is visible while you watch the run.

## Performance

`BarnesHut(theta=0.5)` gives O(N log N) gravity, with measured force errors under 1% (θ = 0.6) and under 0.2% (θ = 0.3), and θ = 0 exact.
One force evaluation for 10,000 stars takes 1.5 s; all-pairs gravity doesn't fit in memory at that size.

Lennard-Jones automatically switches to an O(N) cell list once there are 400 or more particles.
One force evaluation takes 0.044 s for 2,000 atoms (0.36 s with all pairs) and 0.17 s for 8,000 atoms.

## Examples (`examples/`)

- `solar_system.py`: Sun, the inner planets and Jupiter for 20 years. Earth's simulated year is 365.27 days, energy drift is 1e-11, and observed order is 4.0 (expected 4).
- `argon_gas.py`: 216 argon atoms with real Lennard-Jones parameters. Energy is conserved to 2e-5 at dt = 3.2 fs.
- `chemistry.py`: reversible A + B ⇌ C with 0.5 eV of binding energy. The particle simulation gives C = 114 and exact Gillespie kinetics gives C = 111.
- `enzyme.py`: **biology**. Michaelis-Menten E + S ⇌ ES → E + P with diffusing molecules (Brownian dynamics).
  The particle simulation gives P = 242, against 237 ± 9 from exact Gillespie.
- `rahman_argon.py`: **history**. Reproduces Rahman (1964), the first MD simulation of a real liquid: 864 argon atoms at 94.4 K and 1.374 g/cm³. accusim gets **D = (2.54 ± 0.13)e-5 cm²/s** from four 10 ps blocks (Rahman and experiment: 2.43e-5) and a g(r) first peak at 3.75 Å (Rahman ~3.7 Å). Wall time 94 s on a laptop-class CPU. A Green–Kubo cross-check (VACF integral) gives D = 2.44e-5 cm²/s, an independent second route.
- `examples/molten_salt.py`: molten NaCl at 1100 K with cutoff (DSF) electrostatics. The Na–Cl g(r) peak lands at 2.59 Å (experiment 2.6–2.8 Å), and the report's "DSF vs Ewald" row puts the cutoff's force error at ~12% (WARN).
- [EXAMPLES.md](EXAMPLES.md) collects every example's accuracy report, warnings and failures included, with notes on what each grading means.
- `water_cluster.py`: **molecular**. Eight flexible SPC/Fw water molecules (bonds + angles + Lennard-Jones + Coulomb with intramolecular exclusions), 2 ps NVE. Momentum and angular momentum are conserved to 1e-15; energy drift is 4e-4 at dt = 0.2 fs, reported as a WARN.
- `star_cluster.py`: **astronomy**. A 400-star Plummer cluster run for 2 crossing times. The virial ratio goes from 1.04 to 1.02,
  energy drift is 1e-7, and angular momentum holds to 3e-16. Add `--gif` for a rotating 3D animation.
- `fusion.py`: D + T → He-4 + n. The 17.589 MeV comes out of the mass defect automatically, and the neutron gets 14.05 MeV. The report also flags that the neutron's 0.17 c makes Newtonian kinematics about 3% off.

Add `--gif` to write an animation to `examples/out/`.

## Tests

`python -m pytest`. The 86 tests cover:
- **Forces:** F = −∇U is checked numerically for every force (open and periodic), plus Newton's third law.
- **Dynamics against exact answers:** Kepler orbits, Rouse polymer <R²> = (N−1)·3kT/k, Langevin target temperature, energy bounded over 100 eccentric orbits, the measured integrator order, and the exact harmonic-oscillator solution.
- **Electrostatics:** Ewald reproduces the NaCl Madelung constant 1.7475646 (1 and 8 unit cells), and its forces are checked as the gradient of its energy.
- **Engine and models:** the cell list matches the all-pairs results, and JSON models run end to end.
- **Reactions and kinetics:** exponential decay, conservation during reactions, fission energy sharing, the D-T Q-value, endothermic thresholds, Gillespie against the rate equation, and the Brownian diffusion coefficient.

## Limits (honest)

- Direct Gravity, Coulomb and Ewald are O(N²) in memory. Use `BarnesHut` for large gravitating systems. Ewald is meant for a few thousand ions; PME would be the next step.
- Kinematics are Newtonian. The relativity check tells you when that stops being valid.
- Reactions use the Doi contact model; diffusion-limited regimes need a small dt relative to the time particles take to cross the reaction radius.
