"""The accuracy report: which laws should hold, and how well they did.

It decides which conservation laws apply to the configured physics (energy only
for conservative, un-thermostatted systems; momentum only without external
fields or walls; angular momentum only for central forces in open space), then
measures drift, and optionally estimates the time-step error by re-running at
dt/2 and dt/4 and fitting the observed order of convergence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS, WARN, FAIL, NA = "PASS", "WARN", "FAIL", "n/a"


@dataclass
class Check:
    name: str
    status: str
    value: float | None
    detail: str

    def row(self):
        v = "" if self.value is None else f"{self.value:.2e}"
        return f"  [{self.status:>4}] {self.name:<22} {v:>10}  {self.detail}"


def _grade(x, good, ok):
    return PASS if x <= good else WARN if x <= ok else FAIL


class AccuracyReport:
    def __init__(self, checks):
        self.checks = checks

    def __getitem__(self, name) -> Check:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def __str__(self):
        nf = sum(c.status == FAIL for c in self.checks)
        nw = sum(c.status == WARN for c in self.checks)
        if nf:
            head = f"Accuracy report (PROBLEMS FOUND: {nf} failed, {nw} warnings)"
        elif nw:
            head = f"Accuracy report (no failures, {nw} warning{'s' * (nw != 1)})"
        else:
            head = "Accuracy report (all applicable checks passed)"
        return "\n".join([head] + [c.row() for c in self.checks])

    @classmethod
    def build(cls, sim, convergence_duration=None):
        tr, s = sim.trajectory, sim.system
        if len(tr.t) < 2:
            raise ValueError("run the simulation before asking for a report")
        checks = [Check("units", PASS, None, "every input was dimension-checked on entry")]
        integ = sim.integrator
        conservative = all(f.conservative for f in sim.forces) and not integ.thermostat
        external = any(f.external for f in sim.forces)
        walls = s.box is not None and not s.periodic
        central = all(f.central for f in sim.forces)

        E = tr.total_energy
        scale = max(abs(E[0]), np.max(np.abs(tr.kinetic)), np.max(np.abs(tr.potential)), 1e-300)
        if conservative:
            drift = float(np.max(np.abs(E - E[0])) / scale)
            checks.append(Check("energy", _grade(drift, 1e-5, 1e-3), drift,
                                "max |E(t)-E0| / energy scale (incl. reaction ledger)"))
        else:
            why = "thermostat" if integ.thermostat else "dissipative force"
            checks.append(Check("energy", NA, None, f"not conserved by design ({why})"))

        P = np.array(tr.momentum)
        pscale = max(float(np.sum(s.masses[:, None] * np.abs(s.vel))), 1e-300)
        if not external and not walls and not integ.thermostat:
            dp = float(np.max(np.linalg.norm(P - P[0], axis=1)) / pscale)
            checks.append(Check("momentum", _grade(dp, 1e-9, 1e-6), dp, "max |P(t)-P0| / sum m|v|"))
        else:
            checks.append(Check("momentum", NA, None, "external field, walls or thermostat"))

        if central and not external and s.box is None and not integ.thermostat and s.dim in (2, 3):
            L = np.array(tr.angular_momentum)
            lscale = max(float(np.sum(np.linalg.norm(np.cross(s.pos, s.masses[:, None] * s.vel), axis=-1)
                                      if s.dim == 3 else np.abs(np.cross(s.pos, s.masses[:, None] * s.vel)))), 1e-300)
            dl = float(np.max(np.linalg.norm(L - L[0], axis=1)) / lscale)
            checks.append(Check("angular momentum", _grade(dl, 1e-9, 1e-6), dl, "max |L(t)-L0| / sum |r x p|"))
        else:
            checks.append(Check("angular momentum", NA, None, "needs central forces in open space"))

        M = np.array(tr.mass)
        dm = float(np.max(np.abs(M - M[0])) / M[0]) if M[0] else 0.0
        checks.append(Check("mass-energy", _grade(dm, 1e-12, 1e-9), dm, "rest mass + released/c^2"))
        Qc = np.array(tr.charge)
        qscale = max(float(np.sum(np.abs(s.charges))), 1e-300)
        dq = float(np.max(np.abs(Qc - Qc[0])) / qscale)
        checks.append(Check("charge", _grade(dq, 1e-12, 1e-9), dq, "total charge"))

        from .integrators import Langevin
        if isinstance(integ, Langevin) and len(tr.t) >= 4:
            from .units import k_B as KB
            half = len(tr.t) // 2
            kin = np.asarray(tr.kinetic[half:])
            n = np.array([sum(c.values()) for c in tr.counts[half:]], dtype=float)
            T_meas = float(np.mean(2 * kin / (s.dim * np.maximum(n, 1) * KB.value)))
            dev = abs(T_meas - integ.T) / integ.T
            tol = max(0.02, 3 * np.sqrt(2 / (s.dim * max(s.n, 1))))
            checks.append(Check("thermostat", PASS if dev <= tol else (WARN if dev <= 0.1 else FAIL), dev,
                                f"mean T over 2nd half {T_meas:.4g} K vs target {integ.T:.4g} K"))

        from .advisor import timescales
        from .integrators import Brownian
        if not isinstance(integ, Brownian):
            ts = timescales(sim)
            if ts:
                kind, tmin = min(ts.items(), key=lambda kv: kv[1])
                ratio = sim.dt / tmin
                checks.append(Check("timestep vs physics", _grade(ratio, 0.02, 0.1), ratio,
                                    f"dt / shortest timescale ({kind}, {tmin:.3g} s) at end of run"))

        approx = [f for f in sim.forces if not getattr(f, "exact", True)]
        if approx and s.n >= 2:
            from .tree import direct_subset
            sub = np.random.default_rng(0).choice(s.n, size=min(s.n, 200), replace=False)
            errs = []
            for f in approx:
                F_ap = f.compute(s)[0][sub]
                F_ex = direct_subset(f, s, sub)
                errs.append(float(np.sqrt(np.mean(np.sum((F_ap - F_ex) ** 2, 1)) / np.mean(np.sum(F_ex ** 2, 1)))))
            e = max(errs)
            checks.append(Check("force approximation", _grade(e, 1e-3, 1e-2), e,
                                f"RMS relative force error vs exact summation ({len(sub)} sampled particles)"))

        from .forces import Coulomb, DampedShiftedCoulomb, Ewald
        for f in [f for f in sim.forces if isinstance(f, DampedShiftedCoulomb)]:
            q = s.charges
            if not (s.periodic and s.dim == 3 and np.any(q)):
                continue
            if f.exclude or abs(q.sum()) > 1e-9 * np.sum(np.abs(q)) or s.n > 2000:
                checks.append(Check("DSF vs Ewald", NA, None, "needs a neutral system, no exclusions, n <= 2000"))
                continue
            F_d = f.compute(s)[0]
            F_e = Ewald(k=f.k).compute(s)[0]
            e = float(np.sqrt(np.mean(np.sum((F_d - F_e) ** 2, 1)) / np.mean(np.sum(F_e ** 2, 1))))
            checks.append(Check("DSF vs Ewald", _grade(e, 1e-2, 5e-2), e,
                                "RMS force error of the cutoff method vs Ewald, final snapshot (inflated for near-crystalline states)"))
        if s.periodic and any(type(f) is Coulomb for f in sim.forces):
            checks.append(Check("electrostatics", WARN, None,
                                "plain Coulomb in a periodic box uses minimum image (long range truncated): use Ewald"))

        if sim.reactions:
            bi = [r for r in sim.reactions.reactions if len(r.reactants) == 2]
            if bi:
                ldt = max(r.rate for r in bi) * sim.dt
                if ldt > 20:
                    checks.append(Check("reaction timestep", WARN, ldt,
                                        "rate*dt >> 1: reactions fire on first contact (diffusion/contact-limited); "
                                        "the rate constant itself is not resolved"))
                else:
                    checks.append(Check("reaction timestep", _grade(ldt, 0.01, 0.1), ldt,
                                        f"max rate*dt of contact reactions: rates biased low by "
                                        f"~{1 - (1 - np.exp(-ldt)) / ldt:.0%}"))

        from .units import c as C_LIGHT
        beta2 = float(np.max(np.sum(s.vel ** 2, axis=1)) / C_LIGHT.value ** 2) if s.n else 0.0
        checks.append(Check("relativity", _grade(beta2, 1e-4, 1e-2), beta2,
                            f"(v/c)^2 of fastest particle: Newtonian error ~ this size (v = {np.sqrt(beta2):.3g} c)"))

        if convergence_duration is not None:
            checks.append(timestep_convergence(sim, convergence_duration))
        return cls(checks)


def timestep_convergence(sim, duration, start=None) -> Check:
    """Re-run from the initial state at dt, dt/2, dt/4 and estimate the error at dt.

    Uses the observed convergence order p = log2(e1/e2) and Richardson
    extrapolation: err(dt) ~ e1 * 2^p / (2^p - 1).
    """
    from .units import si
    if sim.integrator.stochastic or sim.reactions:
        return Check("timestep error", NA, None, "stochastic run: compare ensemble statistics instead")
    base = start or sim._initial
    T = si(duration, "s")
    n = max(int(round(T / sim.dt)), 1)
    finals = []
    for k in (1, 2, 4):
        c = base.clone(dt=sim.dt / k)
        c.run(steps=n * k, record_every=10 ** 9, positions=False)
        finals.append(c.system.pos.copy())
    size = max(float(np.ptp(finals[2])), 1e-300)
    roundoff = 1e-13 * np.sqrt(n * 4)
    e1 = float(np.sqrt(np.mean((finals[0] - finals[1]) ** 2))) / size
    e2 = float(np.sqrt(np.mean((finals[1] - finals[2]) ** 2))) / size
    expected = sim.integrator.order
    if e1 < 10 * roundoff:
        return Check("timestep error", PASS, e1, "converged to floating-point round-off")
    p = np.log2(e1 / e2)
    est = e1 * 2 ** p / (2 ** p - 1) if p > 0.2 else e1
    if p < 0.5 * expected:
        return Check("timestep error", FAIL if est > 1e-2 else WARN, est,
                     f"observed order {p:.1f} (expected {expected}): chaotic or dt too large; trust statistics, not trajectories")
    return Check("timestep error", _grade(est, 1e-6, 1e-3), est,
                 f"est. position error at dt (relative to system size), observed order {p:.1f}/{expected}")
