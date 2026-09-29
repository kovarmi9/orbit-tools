"""
Mathematical tests of the Lagrange method (positions only, degree n-1).

Generic behaviour (edges, sorting, NaN, errors) is tested for all methods
in test_methods_common.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import STEP, gap_midpoints, interior_queries, monomial, random_polynomials, run
from interp import get_method

LAGRANGE = get_method("lagrange")


# ============================================================
# Polynomial exactness
# ============================================================

@pytest.mark.parametrize("nodes", [2, 3, 4, 6, 8, 10, 12])
def test_reproduces_polynomial_of_its_degree(nodes):
    """n nodes determine a polynomial of degree n-1 exactly."""
    rng = np.random.default_rng(200 + nodes)

    t = np.arange(40) * STEP
    r_true, _ = random_polynomials(rng, nodes - 1, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = interior_queries(t, nodes, rng)

    result = run(LAGRANGE, t, r_true(t), np.zeros((t.size, 3)), q, nodes)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)  # 1 um


@pytest.mark.parametrize("nodes", [2, 3, 6, 10])
def test_error_of_degree_n_monomial(nodes):
    """
    Degree n is NOT reproduced, and the error is known exactly:
    for f(u) = A * u^n the Lagrange error is A * prod_k (u - u_k).
    (Checks that the right nodes are used, not just that the test is sensitive.)
    """
    t = np.arange(nodes + 4) * STEP
    u, r_true, _ = monomial(100.0, nodes, t_center=t.mean())
    q, windows = gap_midpoints(t, nodes)

    result = run(LAGRANGE, t, r_true(t), np.zeros((t.size, 3)), q, nodes)

    expected_error = np.array([-100.0 * np.prod(u(qi) - u(w)) for qi, w in zip(q, windows)])
    assert np.all(np.abs(expected_error) > 1.0)  # m, clearly measurable
    np.testing.assert_allclose(result - r_true(q), np.repeat(expected_error[:, None], 3, axis=1), rtol=1e-6)


def test_irregular_node_spacing():
    rng = np.random.default_rng(207)
    nodes = 10

    t = np.cumsum(rng.uniform(30.0, 90.0, size=40))
    r_true, _ = random_polynomials(rng, nodes - 1, t_center=t.mean(), scale=np.ptp(t) / 2)
    q = interior_queries(t, nodes, rng)

    result = run(LAGRANGE, t, r_true(t), np.zeros((t.size, 3)), q, nodes)

    np.testing.assert_allclose(result, r_true(q), rtol=0, atol=1e-6)


def test_absolute_epoch_with_whole_seconds():
    """Same as for Hermite: epoch ~5.2e9 s, queries shifted by 19 s."""
    rng = np.random.default_rng(211)
    nodes = 10
    t0 = 60310 * 86400.0

    t_rel = np.arange(40) * STEP
    q_rel = t_rel[6:-6] + 19.0
    r_true, _ = random_polynomials(rng, nodes - 1, t_center=t_rel.mean(), scale=np.ptp(t_rel) / 2)
    v = np.zeros((t_rel.size, 3))

    rel = run(LAGRANGE, t_rel, r_true(t_rel), v, q_rel, nodes)
    absolute = run(LAGRANGE, t0 + t_rel, r_true(t_rel), v, t0 + q_rel, nodes)

    np.testing.assert_allclose(absolute, rel, rtol=0, atol=1e-6)


def test_linear_is_midpoint_average():
    """2 nodes = linear interpolation: the gap midpoint is the average of its ends."""
    rng = np.random.default_rng(212)
    t = np.arange(10) * STEP
    r = rng.normal(size=(10, 3)) * 1e6

    result = run(LAGRANGE, t, r, np.zeros((10, 3)), t[:-1] + STEP / 2, 2)

    np.testing.assert_allclose(result, (r[:-1] + r[1:]) / 2, rtol=1e-15)


# ============================================================
# Velocities are ignored
# ============================================================

def test_velocities_are_ignored():
    """Result does not depend on velocities; missing (NaN) velocities are fine."""
    rng = np.random.default_rng(213)
    t = np.arange(30) * STEP
    r = rng.normal(size=(30, 3)) * 1e6
    q = t[8:22] + 17.0

    base = run(LAGRANGE, t, r, np.zeros((30, 3)), q, 10)

    np.testing.assert_array_equal(run(LAGRANGE, t, r, rng.normal(size=(30, 3)), q, 10), base)
    np.testing.assert_array_equal(run(LAGRANGE, t, r, np.full((30, 3), np.nan), q, 10), base)
