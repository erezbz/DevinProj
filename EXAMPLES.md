# Example gallery: what every example's accuracy report says

All examples were run on 2026-10-25 on a 2-vCPU cloud VM (CPU only) from commit `f7a957f`; the Rahman row was re-run at `9e255af`. Each one ends with
`sim.accuracy_report()`. The warnings and failures are left in on purpose: the report is supposed to say when a result shouldn't be trusted.

| Example | Accuracy report | Wall time |
|---|---|---|
| `solar_system.py` | all applicable checks passed | 4.66 s |
| `star_cluster.py` | no failures, 1 warning | 20.75 s |
| `argon_gas.py` | no failures, 2 warnings | 21.74 s |
| `water_cluster.py` | no failures, 2 warnings | 6.61 s |
| `chemistry.py` | all applicable checks passed | 135.11 s |
| `enzyme.py` | no failures, 1 warning | 3.94 s |
| `fusion.py` | PROBLEMS FOUND: 1 failed, 1 warnings | 0.08 s |
| `rigid_water.py` | dt = 2 fs: 2 failed (energy 4.8e-3, timestep); dt = 0.5 fs: energy 2.9e-4 WARN; constraints ≤ 5e-13 PASS | 4.4 s for 3 runs (101 s with Gauss–Seidel SHAKE) |
| `rahman_argon.py` | no failures, 1 warning (energy 4.0e-5) | 33 s (95 s before the Verlet skin) |
| `molten_salt.py` | 1 warning (DSF vs Ewald 0.12–0.14) | 69 s (216 ions), 254 s (512 ions) |

The gradings are only meaningful in context:

- **fusion: relativity FAIL** is real. D–T fusion makes 14.1 MeV neutrons moving at 0.17 c, and the integrator is Newtonian, so their kinetic energy is about 3% off.
  The reaction energy ledger itself is exact (Q-value benchmark 1.8e-7), but the trajectories of the products aren't.
- **fusion / enzyme: reaction timestep.** The fusion contact rate is set to "react on contact" on purpose. In the enzyme model rate*dt = 0.1 means the rates come out about 5% low; halve dt for quantitative kinetics.
- **rigid_water: energy FAIL at 1–2 fs** is real. Making the water rigid only buys ~2.5x in dt here: close O···H Coulomb encounters (~4.5 fs) become the limit, which the timestep row flags.
- **water_cluster: energy 4e-4 + timestep WARN.** The flexible SPC/Fw O–H stretch is fast compared with dt = 0.2 fs. Halving dt to 0.1 fs (checked) cuts the drift 4.1× to 9.9e-5, as expected for a second-order integrator. That is still a WARN; use a smaller dt or more steps for production energies.
- **argon_gas: timestep error WARN.** The observed order is 1.6 instead of 2 (dt comes from `suggest_dt`, capped at 5 fs). My hypothesis, not tested: the steep LJ wall means this dt is still outside the asymptotic regime. The absolute error is 1e-6.
- **molten_salt: DSF vs Ewald WARN.** Cutoff electrostatics with the published settings has about 10% RMS force error (force R² ≈ 0.99). Structure is right
  (Na–Cl peak 2.59 Å, experiment 2.6–2.8 Å), but use `Ewald` if forces must be exact.

Physical results checked against the literature (from the examples' own output):

| Example | Quantity | accusim | Reference |
|---|---|---|---|
| `rahman_argon.py` | self-diffusion, MSD blocks | (2.47 ± 0.10)e-5 cm²/s | Rahman 1964 / experiment: 2.43e-5 |
| `rahman_argon.py` | self-diffusion, Green–Kubo | 2.44e-5 cm²/s | 2.43e-5 |
| `rahman_argon.py` | g(r) first peak | 3.75 Å | ~3.7 Å |
| `molten_salt.py` | Na–Cl g(r) first peak | 2.59 Å | 2.6–2.8 Å |
| `solar_system.py` | energy drift over the run, Yoshida-4 | 1.0e-11 | (exact: 0) |

## Full reports

### `examples/solar_system.py`

```text
advisor chose dt = 0.150 days
Earth's mean sidereal year: 365.27 days (real: 365.26 days, perturbed by planets)
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

### `examples/star_cluster.py`

```text
crossing time 0.75 Myr, virial ratio 2K/|U| at t=0: 1.041
virial ratio after 2 crossing times: 1.022
Accuracy report (no failures, 1 warning)
  [PASS] units                              every input was dimension-checked on entry
  [PASS] energy                   9.55e-08  max |E(t)-E0| / energy scale (incl. reaction ledger)
  [PASS] momentum                 1.91e-16  max |P(t)-P0| / sum m|v|
  [PASS] angular momentum         2.81e-16  max |L(t)-L0| / sum |r x p|
  [PASS] mass-energy              0.00e+00  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [WARN] timestep vs physics      6.18e-02  dt / shortest timescale (closest approach / speed, 9.52e+11 s) at end of run
  [PASS] relativity               2.67e-11  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 5.17e-06 c)
```

### `examples/argon_gas.py`

```text
box 2.429 nm, dt = 3.24 fs
temperature: start 120 K -> end 114.3 K (equilibrating KE <-> PE)
Accuracy report (no failures, 2 warnings)
  [PASS] units                              every input was dimension-checked on entry
  [WARN] energy                   1.73e-05  max |E(t)-E0| / energy scale (incl. reaction ledger)
  [PASS] momentum                 2.10e-16  max |P(t)-P0| / sum m|v|
  [ n/a] angular momentum                   needs central forces in open space
  [PASS] mass-energy              0.00e+00  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [PASS] timestep vs physics      5.66e-03  dt / shortest timescale (closest approach / speed, 5.72e-13 s) at end of run
  [PASS] relativity               3.55e-12  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 1.88e-06 c)
  [WARN] timestep error           1.11e-06  est. position error at dt (relative to system size), observed order 1.6/2
```

### `examples/water_cluster.py`

```text
8 flexible SPC/Fw waters, 2 ps NVE at dt = 0.2 fs
mean O-H length 1.0178 A (SPC/Fw r0 = 1.012 A; thermal stretching makes it slightly longer)
mean H-O-H angle 111.10 deg (theta0 = 113.24 deg; H-H Coulomb is excluded)
Accuracy report (no failures, 2 warnings)
  [PASS] units                              every input was dimension-checked on entry
  [WARN] energy                   4.07e-04  max |E(t)-E0| / energy scale (incl. reaction ledger)
  [PASS] momentum                 1.18e-15  max |P(t)-P0| / sum m|v|
  [PASS] angular momentum         2.95e-15  max |L(t)-L0| / sum |r x p|
  [PASS] mass-energy              0.00e+00  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [WARN] timestep vs physics      4.55e-02  dt / shortest timescale (electrostatic, 4.4e-15 s) at end of run
  [PASS] relativity               2.88e-10  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 1.7e-05 c)
```

### `examples/chemistry.py`

```text
particles: C = 114 after 2.0 us; Gillespie (well-mixed exact): C = 111
reaction events: {'A + B -> C': 532, 'C -> A + B': 418}
Accuracy report (all applicable checks passed)
  [PASS] units                              every input was dimension-checked on entry
  [PASS] energy                   4.73e-15  max |E(t)-E0| / energy scale (incl. reaction ledger)
  [PASS] momentum                 4.12e-17  max |P(t)-P0| / sum m|v|
  [ n/a] angular momentum                   needs central forces in open space
  [PASS] mass-energy              4.72e-16  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [PASS] reaction timestep        4.00e-03  max rate*dt of contact reactions: rates biased low by ~0%
  [PASS] relativity               1.71e-11  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 4.13e-06 c)
```

### `examples/enzyme.py`

```text
diffusion-limited ceiling 4 pi D R = 1.52e-16 m^3/s, k_on used = 4.19e-19 m^3/s
products after 100 ms: particles P = 242, Gillespie P = 236.9 +- 9.4
Accuracy report (no failures, 1 warning)
  [PASS] units                              every input was dimension-checked on entry
  [ n/a] energy                             not conserved by design (thermostat)
  [ n/a] momentum                           external field, walls or thermostat
  [ n/a] angular momentum                   needs central forces in open space
  [PASS] mass-energy              5.25e-16  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [WARN] reaction timestep        1.00e-01  max rate*dt of contact reactions: rates biased low by ~5%
  [PASS] relativity               1.81e-20  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 1.35e-10 c)
```

### `examples/fusion.py`

```text
Q-value from masses: 17.589 MeV (textbook: 17.6 MeV)
   He4: 3.540 MeV, speed 0.044 c
     n: 14.049 MeV, speed 0.173 c
Accuracy report (PROBLEMS FOUND: 1 failed, 1 warnings)
  [PASS] units                              every input was dimension-checked on entry
  [PASS] energy                   0.00e+00  max |E(t)-E0| / energy scale (incl. reaction ledger)
  [PASS] momentum                 1.20e-16  max |P(t)-P0| / sum m|v|
  [PASS] angular momentum         1.53e-16  max |L(t)-L0| / sum |r x p|
  [PASS] mass-energy              0.00e+00  rest mass + released/c^2
  [PASS] charge                   0.00e+00  total charge
  [WARN] reaction timestep        1.00e+06  rate*dt >> 1: reactions fire on first contact (diffusion/contact-limited); the rate constant itself is not resolved
  [FAIL] relativity               2.99e-02  (v/c)^2 of fastest particle: Newtonian error ~ this size (v = 0.173 c)
```
