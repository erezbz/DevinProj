"""Dimensional analysis: quantities carry units so mistakes fail loudly.

Internally everything is SI. A dimension is a vector of exponents over the
base dimensions (mass, length, time, charge, temperature, amount).
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

BASE = ("kg", "m", "s", "C", "K", "mol")


class UnitError(ValueError):
    """Raised when quantities with incompatible dimensions are combined."""


def _dim(*exps) -> tuple:
    exps = list(exps) + [0] * (len(BASE) - len(exps))
    return tuple(Fraction(e) for e in exps)


DIMLESS = _dim()


def dim_str(dim: tuple) -> str:
    parts = []
    for name, e in zip(BASE, dim):
        if e == 0:
            continue
        parts.append(name if e == 1 else f"{name}^{e}")
    return "*".join(parts) or "1"


class Quantity:
    """A value (scalar or numpy array) in SI with a physical dimension."""

    __array_priority__ = 1000

    def __init__(self, value, dim: tuple = DIMLESS):
        self.value = np.asarray(value, dtype=float) if not np.isscalar(value) else float(value)
        self.dim = tuple(Fraction(e) for e in dim)

    def _check_same(self, other, op):
        other = as_quantity(other)
        if other.dim != self.dim:
            raise UnitError(f"cannot {op} [{dim_str(self.dim)}] and [{dim_str(other.dim)}]")
        return other

    def __add__(self, other):
        other = self._check_same(other, "add")
        return Quantity(self.value + other.value, self.dim)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._check_same(other, "subtract")
        return Quantity(self.value - other.value, self.dim)

    def __rsub__(self, other):
        other = self._check_same(other, "subtract")
        return Quantity(other.value - self.value, self.dim)

    def __neg__(self):
        return Quantity(-self.value, self.dim)

    def __mul__(self, other):
        other = as_quantity(other)
        return Quantity(self.value * other.value, tuple(a + b for a, b in zip(self.dim, other.dim)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_quantity(other)
        return Quantity(self.value / other.value, tuple(a - b for a, b in zip(self.dim, other.dim)))

    def __rtruediv__(self, other):
        return as_quantity(other) / self

    def __pow__(self, p):
        p = Fraction(p).limit_denominator(1000)
        return Quantity(self.value ** float(p), tuple(a * p for a in self.dim))

    def sqrt(self):
        return self ** Fraction(1, 2)

    def __lt__(self, other):
        return self.value < self._check_same(other, "compare").value

    def __gt__(self, other):
        return self.value > self._check_same(other, "compare").value

    def __eq__(self, other):
        try:
            other = as_quantity(other)
        except TypeError:
            return NotImplemented
        return self.dim == other.dim and np.all(self.value == other.value)

    def __hash__(self):
        return hash((float(np.sum(self.value)), self.dim))

    def __getitem__(self, idx):
        return Quantity(np.asarray(self.value)[idx], self.dim)

    def __len__(self):
        return len(self.value)

    @property
    def dimensionless(self) -> bool:
        return self.dim == DIMLESS

    def to(self, unit: str):
        """Numeric value expressed in `unit` (raises UnitError on mismatch)."""
        u = parse_unit(unit)
        if u.dim != self.dim:
            raise UnitError(f"cannot convert [{dim_str(self.dim)}] to '{unit}' [{dim_str(u.dim)}]")
        return self.value / u.value

    def si(self, expected: tuple | str | None = None):
        """Numeric SI value, optionally asserting a dimension or unit."""
        if expected is not None:
            exp = parse_unit(expected).dim if isinstance(expected, str) else expected
            if tuple(Fraction(e) for e in exp) != self.dim:
                raise UnitError(f"expected [{dim_str(exp)}], got [{dim_str(self.dim)}]")
        return self.value

    def __repr__(self):
        return f"Quantity({self.value!r}, [{dim_str(self.dim)}])"

    def __format__(self, spec):
        return f"{format(self.value, spec) if np.isscalar(self.value) else self.value} {dim_str(self.dim)}"


def as_quantity(x) -> Quantity:
    if isinstance(x, Quantity):
        return x
    if isinstance(x, (int, float, np.ndarray, list, tuple, np.floating, np.integer)):
        return Quantity(x, DIMLESS)
    raise TypeError(f"not a quantity: {x!r}")


def _u(value, *exps):
    return Quantity(value, _dim(*exps))


# Exact or CODATA 2018 values.
_UNITS: dict[str, Quantity] = {
    "1": _u(1.0),
    "kg": _u(1, 1), "g": _u(1e-3, 1), "mg": _u(1e-6, 1), "t": _u(1e3, 1),
    "u": _u(1.66053906660e-27, 1), "amu": _u(1.66053906660e-27, 1), "Da": _u(1.66053906660e-27, 1),
    "kDa": _u(1.66053906660e-24, 1), "MDa": _u(1.66053906660e-21, 1),
    "Msun": _u(1.98847e30, 1), "Mearth": _u(5.9722e24, 1), "Mjup": _u(1.89813e27, 1),
    "m": _u(1, 0, 1), "km": _u(1e3, 0, 1), "cm": _u(1e-2, 0, 1), "mm": _u(1e-3, 0, 1),
    "um": _u(1e-6, 0, 1), "nm": _u(1e-9, 0, 1), "pm": _u(1e-12, 0, 1), "fm": _u(1e-15, 0, 1),
    "A": _u(1e-10, 0, 1), "angstrom": _u(1e-10, 0, 1),
    "au": _u(1.495978707e11, 0, 1), "ly": _u(9.4607304725808e15, 0, 1), "pc": _u(3.0856775814913673e16, 0, 1),
    "s": _u(1, 0, 0, 1), "ms": _u(1e-3, 0, 0, 1), "us": _u(1e-6, 0, 0, 1), "ns": _u(1e-9, 0, 0, 1),
    "ps": _u(1e-12, 0, 0, 1), "fs": _u(1e-15, 0, 0, 1), "min": _u(60, 0, 0, 1), "h": _u(3600, 0, 0, 1),
    "day": _u(86400, 0, 0, 1), "yr": _u(365.25 * 86400, 0, 0, 1),
    "C": _u(1, 0, 0, 0, 1), "e": _u(1.602176634e-19, 0, 0, 0, 1),
    "K": _u(1, 0, 0, 0, 0, 1),
    "mol": _u(1, 0, 0, 0, 0, 0, 1),
    "N": _u(1, 1, 1, -2), "J": _u(1, 1, 2, -2), "kJ": _u(1e3, 1, 2, -2), "kcal": _u(4184.0, 1, 2, -2),
    "eV": _u(1.602176634e-19, 1, 2, -2), "keV": _u(1.602176634e-16, 1, 2, -2), "MeV": _u(1.602176634e-13, 1, 2, -2),
    "W": _u(1, 1, 2, -3), "Pa": _u(1, 1, -1, -2), "bar": _u(1e5, 1, -1, -2), "atm": _u(101325.0, 1, -1, -2),
    "Hz": _u(1, 0, 0, -1), "L": _u(1e-3, 0, 3), "V": _u(1, 1, 2, -2, -1), "M": _u(1e3, 0, -3, 0, 0, 0, 1),
}

_TOKEN = re.compile(r"\s*([A-Za-z0-9_.]+)(?:\^(-?[0-9]+(?:/[0-9]+)?))?\s*")


def parse_unit(text: str) -> Quantity:
    """Parse strings like 'km/s', 'kg*m^2/s^2', 'kJ/mol', 'm s^-1'."""
    text = text.strip()
    if not text:
        return _UNITS["1"]
    result = Quantity(1.0)
    sign = 1
    pos = 0
    text = text.replace("·", "*")
    while pos < len(text):
        ch = text[pos]
        if ch == "*":
            sign = 1 if sign > 0 else -1
            pos += 1
            continue
        if ch == "/":
            sign = -1
            pos += 1
            continue
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise UnitError(f"cannot parse unit '{text}' at '{text[pos:]}'")
        name, exp = m.group(1), m.group(2)
        if name in _UNITS:
            base = _UNITS[name]
        else:
            try:
                base = Quantity(float(name))
            except ValueError:
                raise UnitError(f"unknown unit '{name}' in '{text}'") from None
        p = Fraction(exp) if exp else Fraction(1)
        result = result * base ** (p * sign)
        pos = m.end()
        if pos < len(text) and text[pos] not in "*/":
            continue
    return result


def Q(value, unit: str = "1") -> Quantity:
    """Construct a quantity: Q(3, 'km/s'), Q([1, 0, 0], 'au')."""
    return as_quantity(value) * parse_unit(unit)


def si(x, unit_or_dim: str | tuple | None = None):
    """SI number from a Quantity, checking dimension. Plain numbers pass through as SI."""
    if isinstance(x, Quantity):
        return x.si(unit_or_dim)
    return np.asarray(x, dtype=float) if not np.isscalar(x) else float(x)


# Physical constants (CODATA 2018).
G = Q(6.67430e-11, "m^3/kg/s^2")
c = Q(299792458.0, "m/s")
k_B = Q(1.380649e-23, "J/K")
N_A = Q(6.02214076e23, "1/mol")
e_charge = Q(1.602176634e-19, "C")
epsilon_0 = Q(8.8541878128e-12, "C^2/J/m")
k_e = 1 / (4 * math.pi * epsilon_0)
