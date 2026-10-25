# DEVLOG: accusim overnight session

All times are UTC (Erez is ~UTC+? local; he said 10:20 PM local at the start).
Rules for this session, set by Erez: 1h30 of code only, then 30 minutes of free time.
No contact with humans and no money spent. I can make my own decisions about the code.

## Phase 1: coding (05:36:54 to 07:06:54 UTC)

- **05:36:54**: Timer started. I recorded the start time to `~/.timer_start`. The repo was empty (no commits).
  Created branch `devin/1761370614-accusim`. Installed `pytest` and `pillow` with `pip --user` (free, from PyPI).
  `git config` was blocked by the harness, which is fine: the default identity is used.
- **05:37**: Design decision: **package name `accusim`**, Python with NumPy only, with all values handled in SI units internally.
  Modules:
  - `units`: dimension vectors with fractions, a unit parser, and CODATA constants.
  - `system`: species plus flat arrays, periodic or reflecting box.
  - `forces`: gravity, Coulomb, Lennard-Jones, harmonic bonds, uniform field, drag.
  - `integrators`: Verlet, Yoshida4, Langevin BAOAB, Brownian.
  - `reactions`: particle reactions plus a Gillespie SSA.
  - `simulation` and `accuracy`: the report.
  - `advisor`: timestep suggestions.
  - `viz`.
- **05:37**: Wrote `units.py`. Smoke check: 1 au/yr = 4.7405 km/s ✔, k_e = 8.98755e9 ✔, and adding m + s raises `UnitError` ✔.
- **05:37–05:38**: Wrote `system.py` and `forces.py`. Smoke check: Lennard-Jones and gravity compute on 5 argon atoms.
- **05:38**: Wrote `integrators.py` and `reactions.py`. Key design choice: reactions conserve momentum exactly by sharing the released
  energy Q = Δm·c² + E_chem in the center-of-mass frame. A reaction below its energy threshold is refused. A single product stores
  the relative kinetic energy as internal energy, which is tracked in a ledger so total energy can be audited.
- **05:38**: Wrote `simulation.py` and `accuracy.py`. The report decides *which* conservation laws should hold for the configured
  physics, then measures how far each one drifted. It also estimates timestep error by re-running at dt/2 and dt/4
  (Richardson extrapolation plus the observed convergence order).
- **05:39**: Wrote `advisor.py`, `viz.py` and `__init__.py`, plus the tests: units, F = −∇U for every force, Newton's 3rd law,
  Kepler orbits, energy over 100 orbits, observed integrator order, harmonic oscillator exact solution, decay law,
  merge conservation, fission energy sharing, D-T fusion 17.6 MeV, endothermic threshold, charge rejection,
  Gillespie vs the rate equation, and the Brownian diffusion coefficient.
