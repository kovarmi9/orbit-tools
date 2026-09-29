from __future__ import annotations

import pytest

from interp import METHODS, get_method
from interp.registry import fixed_degree, register, tied_degree


# ============================================================
# Registry
# ============================================================

def test_hermite_is_registered():
    assert "hermite" in METHODS
    assert METHODS["hermite"].uses_velocity


def test_lagrange_is_registered():
    assert "lagrange" in METHODS
    assert not METHODS["lagrange"].uses_velocity


def test_spline_is_registered():
    assert "spline" in METHODS
    assert not METHODS["spline"].uses_velocity


def test_get_method_is_case_insensitive():
    assert get_method("Hermite") is METHODS["hermite"]


def test_unknown_method_lists_available():
    with pytest.raises(ValueError, match="available: .*hermite"):
        get_method("no-such-method")


def test_duplicate_registration_raises():
    with pytest.raises(RuntimeError, match="registered twice"):
        register("hermite", summary="", uses_velocity=True, resolve=None)(lambda: None)


# ============================================================
# Hermite nodes / degree resolution
# ============================================================

RESOLVE = get_method("hermite").resolve


@pytest.mark.parametrize(
    "nodes, degree, expected",
    [
        (None, None, (6, 11)),   # default
        (4,    None, (4, 7)),
        (None, 7,    (4, 7)),
        (None, 3,    (2, 3)),
        (6,    11,   (6, 11)),   # both given and consistent
    ],
)
def test_hermite_resolve(nodes, degree, expected):
    assert RESOLVE(nodes, degree) == expected


@pytest.mark.parametrize(
    "nodes, degree, message",
    [
        (None, 8,    "odd number >= 3"),
        (None, 1,    "odd number >= 3"),
        (1,    None, "nodes must be >= 2"),
        (4,    11,   "inconsistent"),
    ],
)
def test_hermite_resolve_errors(nodes, degree, message):
    with pytest.raises(ValueError, match=message):
        RESOLVE(nodes, degree)


LAGRANGE_RESOLVE = get_method("lagrange").resolve


@pytest.mark.parametrize(
    "nodes, degree, expected",
    [
        (None, None, (10, 9)),   # default
        (8,    None, (8, 7)),
        (None, 9,    (10, 9)),
        (None, 1,    (2, 1)),    # linear interpolation
        (None, 8,    (9, 8)),    # even degree is fine for Lagrange
    ],
)
def test_lagrange_resolve(nodes, degree, expected):
    assert LAGRANGE_RESOLVE(nodes, degree) == expected


@pytest.mark.parametrize(
    "nodes, degree, message",
    [
        (None, 0,    ">= 1"),
        (1,    None, "nodes must be >= 2"),
        (4,    9,    "inconsistent"),
    ],
)
def test_lagrange_resolve_errors(nodes, degree, message):
    with pytest.raises(ValueError, match=message):
        LAGRANGE_RESOLVE(nodes, degree)


# ============================================================
# Spline (fixed degree, no nodes)
# ============================================================

SPLINE_RESOLVE = get_method("spline").resolve


@pytest.mark.parametrize("degree", [None, 3])
def test_spline_resolve(degree):
    assert SPLINE_RESOLVE(None, degree) == (None, 3)


@pytest.mark.parametrize(
    "nodes, degree, message",
    [
        (6,    None, "uses all data points; --nodes does not apply"),
        (None, 5,    "degree is fixed at 3"),
    ],
)
def test_spline_resolve_errors(nodes, degree, message):
    with pytest.raises(ValueError, match=message):
        SPLINE_RESOLVE(nodes, degree)


def test_fixed_degree_names_the_method():
    with pytest.raises(ValueError, match=r"^akima uses all data points"):
        fixed_degree(3, method="akima")(4, None)


def test_tied_degree_default_is_validated():
    resolve = tied_degree(
        default_nodes=1, min_nodes=2,
        degree_of=lambda n: n - 1, nodes_of=lambda g: g + 1,
        degree_ok=lambda g: g >= 1, degree_rule=">= 1",
    )
    with pytest.raises(ValueError, match="nodes must be >= 2"):
        resolve(None, None)
