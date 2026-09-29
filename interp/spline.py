from __future__ import annotations

import numpy as np

from ._nodes import exact_node_index, prepare
from .registry import fixed_degree, register


# ============================================================
# Cubic spline (not-a-knot end conditions)
# ============================================================

def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for a tridiagonal system, solved for all columns of rhs at once.

    Parameters
    ----------
    lower : ndarray (N,)  - lower[i] multiplies x[i-1] in row i (lower[0] unused)
    diag  : ndarray (N,)
    upper : ndarray (N,)  - upper[i] multiplies x[i+1] in row i (upper[-1] unused)
    rhs   : ndarray (N,k)

    Returns
    -------
    x : ndarray (N,k)
    """
    n = diag.size
    c = np.zeros(n, dtype=float)
    d = np.array(rhs, dtype=float)

    c[0] = upper[0] / diag[0]
    d[0] = d[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / denom
        d[i] = (d[i] - lower[i] * d[i - 1]) / denom

    for i in range(n - 2, -1, -1):
        d[i] -= c[i] * d[i + 1]

    return d


def spline_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    First derivatives at the knots of the C2 cubic spline through (x, y)
    with not-a-knot end conditions (third derivative continuous at x[1] and x[-2]).

    Not-a-knot reproduces any cubic exactly, including at the ends of the data
    (a natural spline would force the second derivative to zero there).

    Parameters
    ----------
    x : ndarray (N,), strictly increasing, N >= 4
    y : ndarray (N,k)

    Returns
    -------
    s : ndarray (N,k)
    """
    h = np.diff(x)                         # (N-1,)
    delta = np.diff(y, axis=0) / h[:, None]  # secant slopes (N-1,k)
    n = x.size

    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros_like(y, dtype=float)

    # interior: continuity of the second derivative at x[i]
    lower[1:-1] = h[1:]
    diag[1:-1] = 2.0 * (h[:-1] + h[1:])
    upper[1:-1] = h[:-1]
    rhs[1:-1] = 3.0 * (h[1:, None] * delta[:-1] + h[:-1, None] * delta[1:])

    # not-a-knot at the start
    w = h[0] + h[1]
    diag[0] = h[1]
    upper[0] = w
    rhs[0] = ((h[0] + 2.0 * w) * h[1] * delta[0] + h[0] ** 2 * delta[1]) / w

    # not-a-knot at the end
    w = h[-1] + h[-2]
    lower[-1] = w
    diag[-1] = h[-2]
    rhs[-1] = (h[-1] ** 2 * delta[-2] + (2.0 * w + h[-1]) * h[-2] * delta[-1]) / w

    return _solve_tridiagonal(lower, diag, upper, rhs)


def spline_evaluate(x: np.ndarray, y: np.ndarray, s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Evaluate the piecewise cubic given by values y and slopes s at the knots x.

    Queries must lie within [x[0], x[-1]].

    Returns
    -------
    ndarray (Q,k)
    """
    i = np.clip(np.searchsorted(x, q, side="right") - 1, 0, x.size - 2)
    h = (x[i + 1] - x[i])[:, None]
    u = ((q - x[i]) / (x[i + 1] - x[i]))[:, None]

    # cubic Hermite basis on [x_i, x_i+1]
    h00 = (1.0 + 2.0 * u) * (1.0 - u) ** 2
    h10 = u * (1.0 - u) ** 2
    h01 = u ** 2 * (3.0 - 2.0 * u)
    h11 = u ** 2 * (u - 1.0)

    return h00 * y[i] + h10 * h * s[i] + h01 * y[i + 1] + h11 * h * s[i + 1]


# ============================================================
# Registered method
# ============================================================

MIN_POINTS = 4  # not-a-knot needs at least 4 knots


@register(
    "spline",
    summary="positions only, global C2 cubic spline over all points (not-a-knot ends)",
    uses_velocity=False,
    resolve=fixed_degree(3, method="spline"),
)
def run(t, r, v, t_query, *, nodes: None, degree: int) -> np.ndarray:
    """Interpolate positions at t_query with a cubic spline through all data points."""
    t, r, v = prepare(t, r, v, uses_velocity=False)

    if t.size < MIN_POINTS:
        raise ValueError(f"Not enough points: need {MIN_POINTS}, got {t.size}.")

    tq = np.asarray(t_query, dtype=float).reshape(-1)
    r_out = np.full((tq.size, 3), np.nan, dtype=float)

    # relative time keeps the spline arithmetic well-conditioned (t ~ 5e9 s)
    t0 = t[0]
    x = t - t0
    s = spline_slopes(x, r)

    inside = (tq >= t[0]) & (tq <= t[-1])
    r_out[inside] = spline_evaluate(x, r, s, tq[inside] - t0)

    # exact node values, same as the local methods
    for i in np.flatnonzero(inside):
        j = exact_node_index(t, float(tq[i]), 1e-12)
        if j is not None:
            r_out[i] = r[j]

    return r_out
