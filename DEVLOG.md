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
- **05:39**: First test run: 23 pass, 1 fail. The decay test used A(10u) → B(20u), which my own physics correctly refused,
  because the mass has to come from somewhere (energy threshold). The test was wrong. I fixed it and added a test that a
  mass increase without energy is forbidden.
- **05:39**: 3 failures to fix:
  1. D-T neutron energy. I expected the textbook 14.07 MeV, but a non-relativistic split for reactants at rest gives 14.049.
     The test now asserts 14.049 and documents why.
  2. The Gillespie test had a single stochastic sample. It now averages 20 runs.
  3. Mass-energy WARN. Chemical energy was being counted as rest mass. I split the ledger into `released`
     (all energy) and `released_rest` (Δm·c² only).
  → **37/37 pass.**
- **05:40**: Committed `81d122e`. Wrote 4 examples: solar system, argon gas, chemistry, fusion.
- **05:41**: Ran the examples, and they found real bugs:
  - **Chemistry:** the particle simulation gave 228 C against 111 from Gillespie, because C → A + B never fired. The bond energy
    stored in C was in a global ledger, not *in the particle*, so the decay couldn't pay its 0.5 eV threshold.
    Fix: a per-particle internal energy (`System.eint`). Merges store energy in the product, and later reactions spend it.
  - **Timestep convergence:** the solar system showed "order −12.5". The dt/2 and dt/4 runs used `round(T/dt)` separately,
    so they ended at different times. Fix: steps = n·k exactly. Added a round-off floor.
  - **New check:** `relativity`, which reports (v/c)² so the user knows when Newtonian mechanics is wrong (the fusion neutron is at 0.17 c).
- **05:43**: Results after the fixes:
  - Chemistry: particles C = 114 against exact Gillespie 111 ✔.
  - Solar system: observed order **4.0/4** ✔, energy drift 1e-11, Earth year 365.27 d.
  - Argon: energy drift 1.7e-5 (WARN at dt = 3.2 fs, which is realistic), observed order 1.6/2 (chaotic system).
  - 39/39 tests pass.
- **05:44**: Wrote README and pyproject.
- **05:44**: Pushed the branch (no PR, as agreed). Remaining plan for the coding window (ends 07:06:54 UTC):
  1. Verify the GIF animation pipeline.
  2. A cell list for short-range Lennard-Jones, so large systems scale O(N).
  3. A declarative JSON model format plus `python -m accusim model.json`, the first step from script toward application.
  4. More tests, then commit and push.
- **05:44**: The GIF pipeline works: `examples/out/solar_system.gif` came out at 406 KB (the output folder is gitignored).
- **05:44**: Added a **cell-list neighbour search** for Lennard-Jones (`forces.cell_list_pairs`), used automatically at 400+ particles.
  New tests check that it matches the all-pairs results (open and periodic boxes, two species) to 1e-10. They pass.
  Benchmark for one force evaluation:

  | Atoms | All pairs | Cell list |
  |---|---|---|
  | 500 | 0.027 s | 0.013 s |
  | 2000 | 0.360 s | 0.044 s |
  | 8000 | skipped (too much memory) | 0.169 s |

  This is linear scaling.
- **05:44**: Added a **declarative JSON model format** (`accusim/model.py`) and a **CLI** (`python -m accusim model.json`).
  Energies can be given in kJ/mol and are converted per molecule.
  - Example model: `examples/models/ozone.json`, toy ozone chemistry with real masses.
  - Run result: O = 112, O2 = 262, O3 = 38; energy drift 3e-15; every check PASS.
  - 43/43 tests pass.
- **05:45**: Added `calibrate_dt(sim, trial_duration, energy_tol)`, which halves dt until a trial run meets an energy tolerance.
  This is "ask for accuracy, not a step size". Test: an eccentric Kepler orbit at a requested 1e-5 tolerance holds over 3 orbits.
  Added a Langevin thermostat test: 125 argon atoms heated from 50 K to a 200 K target land within 8%,
  and the report correctly marks energy "n/a".
- **05:45**: Added `viz.animate3d` (rotating 3D scatter animation) and `--3d` for the CLI, plus a viz smoke test (2D GIF, 3D GIF, energy plot).
- **05:45–05:46**: Added 2 cross-domain examples.
  - `star_cluster.py`: 400-star Plummer sphere (Aarseth-Hénon-Wielen sampling). The virial ratio goes from 1.041 to 1.022 over 2 crossing times,
    energy drift is 9.6e-8, and the momentum and angular-momentum checks are ~2e-16. ✔
  - `enzyme.py`: Michaelis-Menten kinetics with Brownian molecules. It first failed (needed the `kDa` unit, now added).
    Then it **disagreed with Gillespie: 154 against 237**. Diagnosis: with rate·dt = 2, the contact-reaction probability saturates,
    so the effective rate is (1−e^{−λdt})/dt, about 0.43λ. This is exactly the kind of silent error accusim exists to catch, so I added a
    **new report check, `reaction timestep`** (it flags rate·dt and the implied bias), and a test for it.
    With rate·dt = 0.1: **242 against 237 ± 9** ✔. The ozone model had rate·dt = 0.25 (now FAIL), so I set its dt to 10 ps.
  - 47/47 tests pass (I first wrote 48 here by mistake; pytest reported 47).
- **05:47**: Accuracy gap found by reading my own code: plain `Coulomb` in a periodic box uses the minimum image,
  which silently truncates long-range electrostatics. Two fixes:
  1. The report now raises an `electrostatics` WARN in that situation.
  2. New `Ewald` force (real-space erfc + reciprocal k-space + self term), with exact forces.
  Validation: the **NaCl Madelung constant**. The first run gave 1.7475665 against the exact 1.7475646 (1.1e-6), which is precisely the
  erfc(3.5) truncation. I raised alpha to 4/rcut and kmax to 12, and now it matches to 1e-7 for 1 and 8 unit cells,
  with zero net force on lattice ions. Ewald forces are checked as the numerical gradient of its energy. 50/50 tests pass.
- **05:48**: Added a **Barnes-Hut octree** (`accusim/tree.py`). The tree walk is vectorised per node (all particles visiting a node are processed at once).
  Tests: θ = 0 equals direct summation to 1e-12, θ = 0.3 gives force RMS error < 2e-3, θ = 0.6 gives < 1e-2 (1500-star Plummer).
  Benchmark: 2000 stars, 0.15 s (direct 0.18 s); 10,000 stars, 1.48 s (direct doesn't fit in memory).
- **05:48**: Added `viz.export_html`, a **self-contained interactive 3D viewer** (canvas plus about 30 lines of JS, no CDN):
  drag to rotate, scroll to zoom, play/pause, timeline slider. It is also available as the CLI flag `--html`. Smoke test added.
  This is the first step from "script" toward "application". 53/53 tests pass.
- **05:48**: Viewer screenshot check (headless Chrome): it renders the ozone run with legend, controls and slider. A rotated box's corners
  were clipped, so I raised the view extent to 1.75. Added a **`force approximation` report check**: for approximate forces (Barnes-Hut),
  it samples 200 particles and measures the RMS force error against exact summation, so a tree run reports its own approximation error.
  Test added. 54/54 tests pass.
- **05:48**: Added `accusim/io.py`, which saves and loads trajectories (compressed .npz). It supports a particle count that changes from reactions. Round-trip test passes.
- **05:49**: Added **`python -m accusim.validate`**, a scorecard of 8 textbook benchmarks with answers that don't come from accusim:
  Kepler closure (9e-9), oscillator (6e-13), measured Verlet order 2.00008, Madelung (5e-8), D-T Q-value 17.5893 MeV (2e-7),
  decay law (1.1%, 1.2σ stochastic), Gillespie against the rate equation (2e-4), and Einstein diffusion (1.3e-3). All PASS. Wrapped as a test.
- **05:49**: Added a polymer physics test: Rouse chains (harmonic springs, r0 = 0) under Langevin should have <R²> = (N−1)·3kT/k exactly.
  The first attempt **blew up (NaN)**. My dt of 20 ps was larger than the bond vibration period of 57 ps, the classic way people break simulations.
  At dt = 2 ps the test passes within 10%. That suggested a new feature.
- **05:50**: Added a **`timestep vs physics` report check**: dt divided by the shortest physical timescale from the advisor.
  Grades: PASS < 0.02, WARN < 0.1, FAIL above. It is skipped for Brownian dynamics. The first run wrongly FAILed force-free gases,
  because the "closest approach / speed" timescale was being counted when nothing interacts; it now only counts when interaction forces exist.
  Test added. **58/58 tests pass.**
- **05:51**: Re-ran the examples to see what the new check reports. Two problems:
  (1) Fusion showed `reaction timestep` FAIL "biased ~100%". Its contact reaction is deliberately instantaneous (rate·dt = 1e6),
  so it now reports WARN "fires on first contact; rate constant not resolved", and the bias estimate uses the exact 1 − (1 − e^(−x))/x.
  (2) The advisor built a full N×N distance matrix, so a 10k-star Barnes-Hut run would run out of memory, and it ignored Barnes-Hut and softening.
  It now samples ≤1500 rows (O(N) memory), treats Barnes-Hut as gravity, and includes softening in closest approach. Test added for 10k stars.
- **05:52**: 59/59 tests pass. The star cluster still reports `timestep vs physics` WARN (0.062). I kept it: the closest-approach/speed estimate is
  deliberately conservative, and the energy check (9.6e-8) shows the actual run is fine. Both numbers are shown to the user.
- **05:54**: Added **`HarmonicAngle`** (three-body bond-angle springs, analytic gradient; it conserves angular momentum because it is rotationally invariant)
  and **Coulomb `exclude=`** (id pairs with no interaction, so atoms in the same molecule don't attract each other). Both pass the numerical
  F = −∇U checks, periodic and open. The exclusion test checks that exactly that one pair's energy is removed. 61 → 63 tests, all pass.
- **05:53**: Added `examples/water_cluster.py`: 8 flexible SPC/Fw waters with real parameters. Mean O–H 1.0178 Å (r0 1.012), mean H–O–H 111.1°.
  P and L are conserved to ~1e-15, and the energy drift of 4e-4 is a WARN at dt = 0.2 fs. The example exposed a **report bug**: the header said
  "all applicable checks passed" while showing WARN rows. It now reads "no failures, N warnings" or "PROBLEMS FOUND: n failed, m warnings".
  I re-ran the solar system and ozone examples and pasted their real current output into the README. 64/64 tests pass.
- **05:55**: Added `accusim/minimize.py` (**FIRE** energy minimisation) and `accusim/analysis.py` (**g(r)**, **MSD**).
  First run: minimize called `sim.update_forces()`, which returns nothing (it stores the forces on the sim), so the tuple unpack failed. Fixed by using `sim.potential()`.
  The molecule test then missed by 1.2e-6 on the angle: the default stopping force (1e-6 × initial) was too loose, so it is now 1e-9 × initial.
  Tests: LJ dimer at 2^(1/6)σ, distorted triatomic relaxes to r0/θ0, ideal-gas g(r) = 1, FCC first peak at a/√2. **68/68 tests pass.**
- **05:55**: `msd` now unwraps periodic boxes (adds up minimum-image steps between frames). Test: Brownian particles crossing boundaries
  match 6Dt to within 5%.
- **05:56**: Added `examples/rahman_argon.py`, which **reproduces Rahman (1964)**, the first MD study of a real liquid: 864 Ar, 94.4 K, 1.374 g/cm³.
  On a 10 ps run: g(r) peak at **3.75 Å** (Rahman ~3.7 Å) with height 2.75, and **D = 2.15e-5 cm²/s** against Rahman's 2.43e-5, which is 12% low.
  Suspects: statistical noise from one short run, and the cutoff (2.5σ, shifted) differs from Rahman's 2.25σ. Next I'm running a longer production run to separate the two.
  NVE energy drift 4e-5 (WARN), momentum 1e-16. Wall time 36 s.
- **05:58**: A 40 ps single fit gave D = 2.83e-5 and the 10 ps fit gave 2.15e-5, so **statistical scatter dominates**, not the cutoff.
  The example now block-averages: four independent 10 ps blocks, fit for t > 2 ps, mean ± standard error.
- **05:59**: Result: **D = (2.54 ± 0.13)e-5 cm²/s**, within 1σ of Rahman/experiment (2.43e-5). g(r) peak 3.75 Å, height 2.88. Energy drift 4.6e-5 (WARN), 94 s wall time.
  Lesson recorded in the README: give transport coefficients with error bars, never a single fit.
- **06:00**: Added a **`thermostat` report check** for Langevin runs: mean T over the second half against the target, with tolerance max(2%, 3·√(2/dof)).
  Tests: a gas started at 100 K with a 300 K target PASSes; a 1000 K gas with weak friction and a short run FAILs (not equilibrated).
  Added the FIRE LJ-dimer benchmark to `python -m accusim.validate` (9 rows, all PASS, 2e-11). Full suite: **71 passed in 20.00s**.
- **06:01**: Added **`pressure(sim)`**: P = 2K/(dV) − dU/dV, with dU/dV from a central difference of uniform box rescaling. It works for every
  force type, so no per-force virial code is needed. Tests: ideal gas equals NkT/V to 1e-12, and LJ matches an independently computed pairwise virial to 1e-5.
- **06:01**: **Molecules in the JSON format**. A `molecules` section places copies of a template (explicit centres or random) with random
  orientations. It generates bonds and angles (grouped by parameters into HarmonicBond/HarmonicAngle forces) and adds all same-molecule pairs to Coulomb
  `exclude` automatically. New `examples/models/water.json`: 27 SPC/Fw waters. The CLI run conserves P and L to ~1e-15, with energy drift 3e-4 (WARN).
  Test checks counts, bond geometry, angle constant (kJ/mol → J), exclusions, and conservation. Full suite: **74 passed in 21.34s**.
- **06:02**: HTML viewer upgrade. `export_html(..., sim=sim)` now **draws bonds** (when the particle set is fixed) and shows an
  **energy-error sparkline** (relative error over the whole run, with a cursor at the current frame and the max value). The CLI passes `sim`.
  Headless-Chrome screenshot of the 27-water model: molecules drawn with O–H bonds, energy panel reads "max 3.66e-4". Test parses the
  embedded JSON (54 bonds, 21 energy samples). **75/75 tests pass.**
- **06:03**: Quality pass. Checked packaging: `pip install` into a clean venv works and `python -m accusim.validate` passes from the installed copy.
  Ran ruff: no pyflakes errors (no unused names or undefined variables). Applied the safe auto-fixes (import order, stale noqa, sorted `__all__`) and renamed an
  ambiguous variable. The remaining style notes are one-line semicolons in the matplotlib helpers, left alone. Examples re-run OK. Full suite: **75 passed in 22.80s**.
- **06:05**: Reflecting walls now **record the momentum they absorb** (`System.wall_impulse`), and `analysis.wall_pressure(system, duration)` turns
  that into pressure. This is a second, independent pressure route (wall impulse instead of the virial or box rescaling). Test: 2000 argon atoms in a 10×15×20 nm
  reflecting box at 300 K, 100 ps run, with wall pressure checked against kinetic theory 2K/(3V) within 4%. Full suite: **76 passed in 26.80s**.
- **06:07**: New force `DampedShiftedCoulomb` (Fennell & Gezelter 2006 damped shifted force). It is cutoff electrostatics with energy and force
  both going to zero at rcut, a cheaper alternative to Ewald. Bug caught by the gradient test: the first version had the force sign flipped
  (`displacements` returns r_j - r_i). Validation: NaCl 8×8×8 Madelung reads (value, rel. err) = 1.74546 PASS, against Ewald's 5e-8. It is clearly
  labelled *approximate* in the scorecard (now 10 benchmarks, all pass). On a disordered melt it is within 3% of Ewald.
  Full suite: **79 passed in 26.69s**.
