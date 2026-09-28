"""
Mathematical tests of the Hermite method.

Test polynomials are defined in relative time u = (t - t_center) / scale,
so their coefficients stay well-conditioned regardless of the absolute epoch.
"""
from __future__ import annotations

import numpy as np
import pytest

from interp import get_method

HERMITE = get_method("hermite")
STEP = 60.0          # s, typical SP3 sampling
POSITION_SCALE = 7e6  # m, LEO orbit radius


def _random_polynomials(rng, degree: int, t_center: float, scale: float):
    """
    Three random polynomials (one per axis) of the given degree in relative time.

    Returns
    -------
    r(t) : callable -> ndarray (N,3)   position
    v(t) : callable -> ndarray (N,3)   first derivative
    """
    coef = rng.normal(size=(3, degree + 1)) * POSITION_SCALE
    polys = [np.polynomial.Polynomial(c) for c in coef]
    derivs = [p.deriv() for p in polys]

    def r(t):
        u = (np.asarray(t, dtype=float) - t_center) / scale
        return np.column_stack([p(u) for p in polys])

    def v(t):
        u = (np.asarray(t, dtype=float) - t_center) / scale
        return np.column_stack([d(u) for d in derivs]) / scale

    return r, v


def _run(t, r, v, t_query, nodes):
    _, degree = HERMITE.resolve(nodes, None)
    return HERMITE.run(t, r, v, t_query, nodes=nodes, degree=degree)


def _interior_queries(t, nodes, rng, count=50):
    """Random query times that have enough nodes on both sides (and are not nodes)."""
    left = nodes // 2
    right = nodes - left
    lo, hi = t[left - 1], t[len(t) - right]
    q = rng.uniform(lo, hi, size=count)
    return q[np.min(np.abs(q[:, None] - t[None, :]), axis=1) > 1e-3]


# ============================================================
# Polynomial exactness
# ============================================================

@pytest.mark.parametrize("nodes", [2, 3, 4, 5, 6, 7, 8])
def test_reproduces_polynomial_of_its_degree(nodes):
    """n nodes with values + derivatives determine a polynomial of degree 2n-1 exactly."""
    rng = np.random.default_rng(nodes)
    degree = 2 * nodes - 1

    t = np.arange(40) * STEP
    r_true, v_true = _random_polynomials(rng, degree, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = _interior_queries(t, nodes, rng)

    result = _run(t, r_true(t), v_true(t), q, nodes)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)  # 1 um


@pytest.mark.parametrize("nodes", [2, 4, 6])
def test_error_of_degree_2n_monomial(nodes):
    """
    Degree 2n is NOT reproduced, and the error is known exactly:
    for f(u) = A * u^(2n) the Hermite error is A * prod_k (u - u_k)^2.
    (Checks that the right nodes are used, not just that the test is sensitive.)
    """
    amplitude = 100.0  # m
    t = np.arange(2 * nodes + 2) * STEP
    t_center = t.mean()

    def u(x):
        return (np.asarray(x, dtype=float) - t_center) / STEP

    def r_true(x):
        return np.repeat((amplitude * u(x) ** (2 * nodes))[:, None], 3, axis=1)

    def v_true(x):
        return np.repeat((amplitude * 2 * nodes * u(x) ** (2 * nodes - 1) / STEP)[:, None], 3, axis=1)

    left = nodes // 2
    right = nodes - left
    gaps = np.arange(left, len(t) - right + 1)          # computable gaps
    q = (t[gaps - 1] + t[gaps]) / 2                     # middle of each gap

    result = _run(t, r_true(t), v_true(t), q, nodes)

    expected_error = np.empty(q.size)
    for i, g in enumerate(gaps):
        window = u(t[g - left : g + right])
        expected_error[i] = -amplitude * np.prod((u(q[i]) - window) ** 2)

    assert np.all(np.abs(expected_error) > 1.0)  # m, clearly measurable
    np.testing.assert_allclose(result - r_true(q), np.repeat(expected_error[:, None], 3, axis=1), rtol=1e-6)


def test_irregular_node_spacing():
    """Hermite does not require equidistant nodes."""
    rng = np.random.default_rng(7)
    nodes = 6

    t = np.cumsum(rng.uniform(30.0, 90.0, size=40))
    r_true, v_true = _random_polynomials(rng, 2 * nodes - 1, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = _interior_queries(t, nodes, rng)

    result = _run(t, r_true(t), v_true(t), q, nodes)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)


def test_absolute_epoch_with_whole_seconds():
    """
    Realistic magnitude: seconds since MJD epoch (~5.2e9 s), nodes on whole seconds,
    queries shifted by 19 s (TAI vs GPS). Must give the same result as relative time.
    """
    rng = np.random.default_rng(11)
    nodes = 6
    t0 = 60310 * 86400.0

    t_rel = np.arange(40) * STEP
    q_rel = t_rel[5:-5] + 19.0
    r_true, v_true = _random_polynomials(rng, 2 * nodes - 1, t_center=t_rel.mean(), scale=np.ptp(t_rel) / 2)

    rel = _run(t_rel, r_true(t_rel), v_true(t_rel), q_rel, nodes)
    absolute = _run(t0 + t_rel, r_true(t_rel), v_true(t_rel), t0 + q_rel, nodes)

    np.testing.assert_allclose(absolute, rel, rtol=0, atol=1e-6)


# ============================================================
# Nodes, edges and input handling
# ============================================================

def test_exact_node_returns_node_value():
    rng = np.random.default_rng(3)
    t = np.arange(20) * STEP
    r = rng.normal(size=(20, 3)) * POSITION_SCALE
    v = rng.normal(size=(20, 3)) * 7e3

    result = _run(t, r, v, t[[0, 7, 19]], nodes=6)

    np.testing.assert_array_equal(result, r[[0, 7, 19]])


@pytest.mark.parametrize("nodes", [2, 5, 6])
def test_outside_range_gives_nan(nodes):
    """Queries without enough nodes on both sides are NaN, all others are computed."""
    t = np.arange(20) * STEP
    r = np.ones((20, 3))
    v = np.zeros((20, 3))
    left = nodes // 2
    right = nodes - left

    q = np.array([
        t[0] - 30.0,                     # before data
        t[left - 1] - 30.0,              # not enough nodes on the left (if left > 1)
        t[left - 1] + 30.0,              # first computable gap
        t[len(t) - right] - 30.0,        # last computable gap
        t[len(t) - right] + 30.0,        # not enough nodes on the right (if right > 1)
        t[-1] + 30.0,                    # after data
    ])
    result = _run(t, r, v, q, nodes)
    computed = np.isfinite(result).all(axis=1)

    assert computed.tolist() == [False, False, True, True, False, False]


def test_unsorted_input_gives_same_result():
    rng = np.random.default_rng(5)
    t = np.arange(30) * STEP
    r = rng.normal(size=(30, 3))
    v = rng.normal(size=(30, 3))
    q = t[10:20] + 17.0
    perm = rng.permutation(30)

    np.testing.assert_array_equal(_run(t[perm], r[perm], v[perm], q, 6), _run(t, r, v, q, 6))


def test_nan_rows_are_dropped():
    """A NaN row behaves as if the epoch were missing."""
    rng = np.random.default_rng(6)
    t = np.arange(30) * STEP
    r = rng.normal(size=(30, 3))
    v = rng.normal(size=(30, 3))
    q = t[10:20] + 17.0

    r_nan = r.copy()
    r_nan[15] = np.nan
    keep = np.arange(30) != 15

    np.testing.assert_array_equal(_run(t, r_nan, v, q, 6), _run(t[keep], r[keep], v[keep], q, 6))


def test_duplicate_times_raise():
    t = np.array([0.0, 60.0, 60.0, 120.0, 180.0, 240.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(t, np.zeros((6, 3)), np.zeros((6, 3)), [90.0], nodes=2)


def test_not_enough_points_raise():
    t = np.arange(4) * STEP
    with pytest.raises(ValueError, match="Not enough points"):
        _run(t, np.zeros((4, 3)), np.zeros((4, 3)), [90.0], nodes=6)
