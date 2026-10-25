"""accusim: an accuracy-first universal particle simulator.

Particles of any kind (atoms, molecules, cells, planets) with gravity,
electrostatics, Lennard-Jones, bonds, fields and drag, plus reactions that
conserve momentum, charge and mass-energy exactly, and a report on how
accurate each run was.
"""
from .accuracy import AccuracyReport
from .analysis import msd, rdf
from .minimize import minimize
from .advisor import calibrate_dt, suggest_dt, timescales
from .forces import Coulomb, Drag, Ewald, Gravity, HarmonicAngle, HarmonicBond, LennardJones, UniformField
from .integrators import Brownian, Langevin, VelocityVerlet, Yoshida4
from .reactions import Reaction, gillespie
from .simulation import Simulation, Trajectory
from .system import Species, System
from .tree import BarnesHut
from .units import G, N_A, Q, Quantity, UnitError, c, k_B, k_e, parse_unit

__all__ = [
    "AccuracyReport", "BarnesHut", "calibrate_dt", "Brownian", "Coulomb", "Drag", "Ewald", "G", "Gravity", "HarmonicAngle", "HarmonicBond", "Langevin",
    "LennardJones", "N_A", "Q", "Quantity", "Reaction", "Simulation", "Species", "System",
    "Trajectory", "UniformField", "UnitError", "VelocityVerlet", "Yoshida4", "c", "gillespie",
    "k_B", "k_e", "minimize", "msd", "parse_unit", "rdf", "suggest_dt", "timescales",
]
__version__ = "0.1.0"
