"""accusim: an accuracy-first universal particle simulator.

Particles of any kind (atoms, molecules, cells, planets) with gravity,
electrostatics, Lennard-Jones, bonds, fields and drag, plus reactions that
conserve momentum, charge and mass-energy exactly, and a report on how
accurate each run was.
"""
from .accuracy import AccuracyReport
from .advisor import calibrate_dt, suggest_dt, timescales
from .analysis import msd, pressure, rdf
from .forces import Coulomb, Drag, Ewald, DampedShiftedCoulomb, Gravity, HarmonicAngle, HarmonicBond, LennardJones, UniformField
from .integrators import Brownian, Langevin, NoseHoover, Rattle, VelocityVerlet, Yoshida4
from .minimize import minimize
from .reactions import Reaction, gillespie
from .simulation import Simulation, Trajectory
from .system import Species, System
from .tree import BarnesHut
from .units import N_A, G, Q, Quantity, UnitError, c, k_B, k_e, parse_unit

__all__ = [
    "N_A",
    "AccuracyReport",
    "BarnesHut",
    "Brownian",
    "Coulomb",
    "DampedShiftedCoulomb",
    "Drag",
    "Ewald",
    "G",
    "Gravity",
    "HarmonicAngle",
    "HarmonicBond",
    "Langevin",
    "LennardJones",
    "NoseHoover",
    "Q",
    "Quantity",
    "Rattle",
    "Reaction",
    "Simulation",
    "Species",
    "System",
    "Trajectory",
    "UniformField",
    "UnitError",
    "VelocityVerlet",
    "Yoshida4",
    "c",
    "calibrate_dt",
    "gillespie",
    "k_B",
    "k_e",
    "minimize",
    "msd",
    "parse_unit",
    "pressure",
    "rdf",
    "suggest_dt",
    "timescales",
]
__version__ = "0.1.0"
