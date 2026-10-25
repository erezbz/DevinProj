"""accusim: an accuracy-first universal particle simulator.

Particles of any kind (atoms, molecules, cells, planets) with gravity,
electrostatics, Lennard-Jones, bonds, fields and drag, plus reactions that
conserve momentum, charge and mass-energy exactly, and a report on how
accurate each run was.
"""
from .accuracy import AccuracyReport
from .advisor import calibrate_dt, suggest_dt, timescales
from .forces import Coulomb, Drag, Ewald, Gravity, HarmonicBond, LennardJones, UniformField
from .integrators import Brownian, Langevin, VelocityVerlet, Yoshida4
from .reactions import Reaction, gillespie
from .simulation import Simulation, Trajectory
from .system import Species, System
from .units import G, N_A, Q, Quantity, UnitError, c, k_B, k_e, parse_unit

__all__ = [
    "AccuracyReport", "calibrate_dt", "Brownian", "Coulomb", "Drag", "Ewald", "G", "Gravity", "HarmonicBond", "Langevin",
    "LennardJones", "N_A", "Q", "Quantity", "Reaction", "Simulation", "Species", "System",
    "Trajectory", "UniformField", "UnitError", "VelocityVerlet", "Yoshida4", "c", "gillespie",
    "k_B", "k_e", "parse_unit", "suggest_dt", "timescales",
]
__version__ = "0.1.0"
