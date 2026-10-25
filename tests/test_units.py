import numpy as np
import pytest

from accusim import Q, UnitError, parse_unit
from accusim.units import G


def test_conversion_and_arithmetic():
    assert Q(3, "km/s").to("m/s") == pytest.approx(3000)
    assert (Q(1, "au") / Q(1, "yr")).to("km/s") == pytest.approx(4.7405, rel=1e-4)
    assert (Q(2, "m") * Q(3, "s")).to("m*s") == pytest.approx(6)
    assert Q(1, "kJ/mol").to("J/mol") == pytest.approx(1000)


def test_mismatched_dimensions_fail_loudly():
    with pytest.raises(UnitError):
        Q(1, "m") + Q(1, "s")
    with pytest.raises(UnitError):
        Q(1, "kg").to("m")


def test_parser_variants():
    assert parse_unit("m s^-1").dim == parse_unit("m/s").dim
    assert parse_unit("kg*m^2/s^2").dim == parse_unit("J").dim
    with pytest.raises(UnitError):
        parse_unit("furlong")


def test_surface_gravity_of_earth():
    g = G * Q(1, "Mearth") / Q(6371, "km") ** 2
    assert g.to("m/s^2") == pytest.approx(9.82, rel=1e-2)


def test_array_quantities():
    v = Q([1, 2, 3], "km")
    assert np.allclose(v.to("m"), [1000, 2000, 3000])
