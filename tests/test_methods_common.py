"""
Behaviour every registered method must have (edges, sorting, NaN rows, errors).

Tests are parametrized over interp.METHODS, so a newly registered method
is covered automatically. Local (sliding-window) methods run with 6 nodes,
global methods (e.g. spline) with their only setting.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import POSITION_SCALE, STEP, is_local, run, window_split
from interp import METHODS

LOCAL = [m for m in METHODS.values() if is_local(m)]
GLOBAL = [m for m in METHODS.values() if not is_local(m)]


def _params(methods):
    return pytest.mark.parametrize("method", methods, ids=[m.name for m in methods])


ALL_METHODS = _params(list(METHODS.values()))


def _nodes(method) -> int | None:
    return 6 if is_local(method) else None


def _random_orbit(seed: int, n: int):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * STEP
    r = rng.normal(size=(n, 3)) * POSITION_SCALE
    v = rng.normal(size=(n, 3)) * 7e3
    return rng, t, r, v


@ALL_METHODS
def test_exact_node_returns_node_value(method):
    """A query exactly at a node returns that node - also at the very edges of the data."""
    _, t, r, v = _random_orbit(3, 20)

    result = run(method, t, r, v, t[[0, 7, 19]], _nodes(method))

    np.testing.assert_array_equal(result, r[[0, 7, 19]])


@_params(LOCAL)
@pytest.mark.parametrize("nodes", [2, 5, 6])
def test_outside_range_gives_nan_local(method, nodes):
    """Queries without enough nodes on both sides are NaN, all others are computed."""
    t = np.arange(20) * STEP
    r = np.ones((20, 3))
    v = np.zeros((20, 3))
    left, right = window_split(nodes)

    q = np.array([
        t[0] - 30.0,                     # before data
        t[left - 1] - 30.0,              # not enough nodes on the left (if left > 1)
        t[left - 1] + 30.0,              # first computable gap
        t[len(t) - right] - 30.0,        # last computable gap
        t[len(t) - right] + 30.0,        # not enough nodes on the right (if right > 1)
        t[-1] + 30.0,                    # after data
    ])
    result = run(method, t, r, v, q, nodes)
    computed = np.isfinite(result).all(axis=1)

    assert computed.tolist() == [False, False, True, True, False, False]


@_params(GLOBAL)
def test_outside_range_gives_nan_global(method):
    """Global methods cover the whole data span, including the first and last gap."""
    t = np.arange(20) * STEP
    r = np.ones((20, 3))
    v = np.zeros((20, 3))

    q = np.array([t[0] - 30.0, t[0] + 30.0, t[-1] - 30.0, t[-1] + 30.0])
    computed = np.isfinite(run(method, t, r, v, q)).all(axis=1)

    assert computed.tolist() == [False, True, True, False]


@ALL_METHODS
def test_constant_orbit_is_reproduced(method):
    """Constant position (zero velocity) is trivially reproduced between nodes."""
    t = np.arange(20) * STEP
    r = np.tile([7e6, -3e6, 1e6], (20, 1))

    result = run(method, t, r, np.zeros((20, 3)), t[6:14] + 23.0, _nodes(method))

    np.testing.assert_allclose(result, r[:8], rtol=1e-12)


@ALL_METHODS
def test_unsorted_input_gives_same_result(method):
    rng, t, r, v = _random_orbit(5, 30)
    q = t[10:20] + 17.0
    perm = rng.permutation(30)

    np.testing.assert_array_equal(
        run(method, t[perm], r[perm], v[perm], q, _nodes(method)),
        run(method, t, r, v, q, _nodes(method)),
    )


@ALL_METHODS
def test_nan_position_row_is_dropped(method):
    """A row with NaN position behaves as if the epoch were missing."""
    _, t, r, v = _random_orbit(6, 30)
    q = t[10:20] + 17.0

    r_nan = r.copy()
    r_nan[15] = np.nan
    keep = np.arange(30) != 15

    np.testing.assert_array_equal(
        run(method, t, r_nan, v, q, _nodes(method)),
        run(method, t[keep], r[keep], v[keep], q, _nodes(method)),
    )


@ALL_METHODS
def test_duplicate_times_raise(method):
    t = np.array([0.0, 60.0, 60.0, 120.0, 180.0, 240.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        run(method, t, np.zeros((6, 3)), np.zeros((6, 3)), [90.0], _nodes(method))


@ALL_METHODS
def test_not_enough_points_raise(method):
    t = np.arange(3) * STEP
    with pytest.raises(ValueError, match="Not enough points"):
        run(method, t, np.zeros((3, 3)), np.zeros((3, 3)), [90.0], _nodes(method))


@ALL_METHODS
def test_default_settings_are_valid(method):
    """The method's default nodes/degree pass its own validation and are consistent."""
    nodes, degree = method.resolve(None, None)

    assert method.resolve(nodes, None) == (nodes, degree)
    assert method.resolve(None, degree) == (nodes, degree)
