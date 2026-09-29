"""
Mathematical tests of the Hermite method (positions + velocities, degree 2n-1).

Generic behaviour (edges, sorting, NaN, errors) is tested for all methods
in test_methods_common.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import STEP, gap_midpoints, interior_queries, monomial, random_polynomials, run
from interp import get_method

HERMITE = get_method("hermite")


# ============================================================
# Polynomial exactness
# ============================================================

@pytest.mark.parametrize("nodes", [2, 3, 4, 5, 6, 7, 8])
def test_reproduces_polynomial_of_its_degree(nodes):
    """n nodes with values + derivatives determine a polynomial of degree 2n-1 exactly."""
    rng = np.random.default_rng(nodes)
    degree = 2 * nodes - 1

    t = np.arange(40) * STEP
    r_true, v_true = random_polynomials(rng, degree, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = interior_queries(t, nodes, rng)

    result = run(HERMITE, t, r_true(t), v_true(t), q, nodes)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)  # 1 um


@pytest.mark.parametrize("nodes", [2, 4, 6])
def test_error_of_degree_2n_monomial(nodes):
    """
    Degree 2n is NOT reproduced, and the error is known exactly:
    for f(u) = A * u^(2n) the Hermite error is A * prod_k (u - u_k)^2.
    (Checks that the right nodes are used, not just that the test is sensitive.)
    """
    t = np.arange(2 * nodes + 2) * STEP
    u, r_true, v_true = monomial(100.0, 2 * nodes, t_center=t.mean())
    q, windows = gap_midpoints(t, nodes)

    result = run(HERMITE, t, r_true(t), v_true(t), q, nodes)

    expected_error = np.array([-100.0 * np.prod((u(qi) - u(w)) ** 2) for qi, w in zip(q, windows)])
    assert np.all(np.abs(expected_error) > 1.0)  # m, clearly measurable
    np.testing.assert_allclose(result - r_true(q), np.repeat(expected_error[:, None], 3, axis=1), rtol=1e-6)


def test_irregular_node_spacing():
    """Hermite does not require equidistant nodes."""
    rng = np.random.default_rng(7)
    nodes = 6

    t = np.cumsum(rng.uniform(30.0, 90.0, size=40))
    r_true, v_true = random_polynomials(rng, 2 * nodes - 1, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = interior_queries(t, nodes, rng)

    result = run(HERMITE, t, r_true(t), v_true(t), q, nodes)

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
    r_true, v_true = random_polynomials(rng, 2 * nodes - 1, t_center=t_rel.mean(), scale=np.ptp(t_rel) / 2)

    rel = run(HERMITE, t_rel, r_true(t_rel), v_true(t_rel), q_rel, nodes)
    absolute = run(HERMITE, t0 + t_rel, r_true(t_rel), v_true(t_rel), t0 + q_rel, nodes)

    np.testing.assert_allclose(absolute, rel, rtol=0, atol=1e-6)


# ============================================================
# Velocities
# ============================================================

def test_velocities_are_used():
    """Changing only the velocities changes the result (between nodes)."""
    rng = np.random.default_rng(12)
    t = np.arange(20) * STEP
    r = rng.normal(size=(20, 3)) * 1e3
    v = rng.normal(size=(20, 3))
    q = t[5:15] + 30.0

    # random change: a constant one would cancel out at the midpoint of a symmetric window
    changed = run(HERMITE, t, r, v + rng.normal(size=(20, 3)) * 100.0, q, 6)

    assert np.all(np.abs(changed - run(HERMITE, t, r, v, q, 6)) > 1.0)


def test_nan_velocity_drops_row():
    """A row with NaN velocity cannot be used by Hermite and behaves as a missing epoch."""
    rng = np.random.default_rng(13)
    t = np.arange(30) * STEP
    r = rng.normal(size=(30, 3))
    v = rng.normal(size=(30, 3))
    q = t[10:20] + 17.0

    v_nan = v.copy()
    v_nan[15] = np.nan
    keep = np.arange(30) != 15

    np.testing.assert_array_equal(run(HERMITE, t, r, v_nan, q, 6), run(HERMITE, t[keep], r[keep], v[keep], q, 6))
