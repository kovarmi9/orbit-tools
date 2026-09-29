"""
Mathematical tests of the cubic spline (positions only, global, not-a-knot ends).

Generic behaviour (edges, sorting, NaN, errors) is tested for all methods
in test_methods_common.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import POSITION_SCALE, STEP, random_polynomials, run
from interp import get_method
from interp.spline import spline_evaluate, spline_slopes

SPLINE = get_method("spline")


def _whole_span_queries(t, rng, count=200):
    """Random queries over the whole data span, including the first and last gap."""
    return rng.uniform(t[0], t[-1], size=count)


# ============================================================
# Exactness and convergence
# ============================================================

@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_reproduces_cubics_everywhere(degree):
    """
    Not-a-knot spline reproduces any polynomial up to degree 3 exactly,
    also in the first and last gap (a natural spline would not).
    """
    rng = np.random.default_rng(300 + degree)
    t = np.arange(40) * STEP
    r_true, _ = random_polynomials(rng, degree, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = _whole_span_queries(t, rng)

    result = run(SPLINE, t, r_true(t), np.zeros((t.size, 3)), q)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)  # 1 um


def test_does_not_reproduce_quartic():
    """Sanity check of the test above: degree 4 is NOT reproduced."""
    rng = np.random.default_rng(304)
    t = np.arange(40) * STEP
    r_true, _ = random_polynomials(rng, 4, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = _whole_span_queries(t, rng)

    result = run(SPLINE, t, r_true(t), np.zeros((t.size, 3)), q)

    assert np.max(np.abs(result - r_true(q))) > 1.0  # m


def test_irregular_node_spacing():
    rng = np.random.default_rng(307)
    t = np.cumsum(rng.uniform(30.0, 90.0, size=40))
    r_true, _ = random_polynomials(rng, 3, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = _whole_span_queries(t, rng)

    result = run(SPLINE, t, r_true(t), np.zeros((t.size, 3)), q)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)


def test_fourth_order_convergence():
    """
    For a smooth circular orbit the interior error falls with h^4:
    halving the step reduces it about 16x (a bug usually drops the order to h^2 or h).
    """
    omega = 2 * np.pi / 5900.0  # LEO orbital period ~98 min

    def orbit(t):
        return POSITION_SCALE * np.column_stack([np.cos(omega * t), np.sin(omega * t), np.zeros_like(t)])

    def max_error(step):
        t = np.arange(0.0, 6 * 3600.0 + step / 2, step)
        q = (t[:-1] + t[1:])[20:-20] / 2  # gap midpoints, away from the ends
        result = run(SPLINE, t, orbit(t), np.zeros((t.size, 3)), q)
        return np.max(np.linalg.norm(result - orbit(q), axis=1))

    ratio = max_error(240.0) / max_error(120.0)

    assert 14.0 < ratio < 18.0


def test_absolute_epoch_with_whole_seconds():
    """Epoch ~5.2e9 s, queries shifted by 19 s: same as relative time."""
    rng = np.random.default_rng(311)
    t0 = 60310 * 86400.0

    t_rel = np.arange(40) * STEP
    q_rel = t_rel[:-1] + 19.0
    r_true, _ = random_polynomials(rng, 3, t_center=t_rel.mean(), scale=np.ptp(t_rel) / 2)
    v = np.zeros((t_rel.size, 3))

    rel = run(SPLINE, t_rel, r_true(t_rel), v, q_rel)
    absolute = run(SPLINE, t0 + t_rel, r_true(t_rel), v, t0 + q_rel)

    np.testing.assert_allclose(absolute, rel, rtol=0, atol=1e-6)


# ============================================================
# Spline properties
# ============================================================

def test_second_derivative_is_continuous():
    """C2: second derivative from the left and right of every interior knot agrees."""
    rng = np.random.default_rng(320)
    x = np.cumsum(rng.uniform(30.0, 90.0, size=30))
    y = rng.normal(size=(30, 3)) * 1e3
    s = spline_slopes(x, y)

    h = np.diff(x)[:, None]
    delta = np.diff(y, axis=0) / h
    # second derivative of the cubic Hermite piece at its left / right end
    left_end = (6 * delta - 4 * s[:-1] - 2 * s[1:]) / h
    right_end = (-6 * delta + 2 * s[:-1] + 4 * s[1:]) / h

    np.testing.assert_allclose(right_end[:-1], left_end[1:], rtol=1e-9, atol=1e-12)


def test_evaluate_passes_through_knots():
    rng = np.random.default_rng(321)
    x = np.cumsum(rng.uniform(30.0, 90.0, size=20))
    y = rng.normal(size=(20, 3)) * POSITION_SCALE

    np.testing.assert_allclose(spline_evaluate(x, y, spline_slopes(x, y), x), y, rtol=1e-14)


def test_matches_scipy():
    """Independent check against scipy's CubicSpline (not a runtime dependency)."""
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(322)
    t = np.cumsum(rng.uniform(30.0, 90.0, size=200))
    r = rng.normal(size=(200, 3)) * POSITION_SCALE
    q = _whole_span_queries(t, rng, count=1000)

    expected = scipy_interpolate.CubicSpline(t, r, bc_type="not-a-knot")(q)

    np.testing.assert_allclose(run(SPLINE, t, r, np.zeros((200, 3)), q), expected, rtol=0, atol=1e-6)


def test_velocities_are_ignored():
    rng = np.random.default_rng(323)
    t = np.arange(30) * STEP
    r = rng.normal(size=(30, 3)) * 1e6
    q = t[:-1] + 17.0

    base = run(SPLINE, t, r, np.zeros((30, 3)), q)

    np.testing.assert_array_equal(run(SPLINE, t, r, rng.normal(size=(30, 3)), q), base)
    np.testing.assert_array_equal(run(SPLINE, t, r, np.full((30, 3), np.nan), q), base)


def test_minimum_four_points():
    t = np.arange(4) * STEP
    r = np.arange(12, dtype=float).reshape(4, 3)

    result = run(SPLINE, t, r, np.zeros((4, 3)), [90.0])

    assert np.isfinite(result).all()
