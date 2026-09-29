"""
Shared helpers for the interpolation method tests.

Test polynomials are defined in relative time u = (t - t_center) / scale,
so their coefficients stay well-conditioned regardless of the absolute epoch.
"""
from __future__ import annotations

import numpy as np

STEP = 60.0           # s, typical SP3 sampling
POSITION_SCALE = 7e6  # m, LEO orbit radius


def random_polynomials(rng, degree: int, t_center: float, scale: float):
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


def monomial(amplitude: float, power: int, t_center: float, scale: float = STEP):
    """
    f(u) = amplitude * u^power on all three axes, u = (t - t_center) / scale.

    Returns
    -------
    u(t), r(t), v(t) : callables
    """
    def u(t):
        return (np.asarray(t, dtype=float) - t_center) / scale

    def r(t):
        return np.repeat((amplitude * u(t) ** power)[:, None], 3, axis=1)

    def v(t):
        return np.repeat((amplitude * power * u(t) ** (power - 1) / scale)[:, None], 3, axis=1)

    return u, r, v


def run(method, t, r, v, t_query, nodes: int | None = None) -> np.ndarray:
    """Run a registered method with the given number of nodes (None = method default)."""
    nodes, degree = method.resolve(nodes, None)
    return method.run(t, r, v, t_query, nodes=nodes, degree=degree)


def is_local(method) -> bool:
    """True for sliding-window methods, False for global ones (e.g. spline)."""
    nodes, _ = method.resolve(None, None)
    return nodes is not None


def window_split(nodes: int) -> tuple[int, int]:
    """Nodes on the left / right of the query (see interp._nodes.select_nodes)."""
    left = nodes // 2
    return left, nodes - left


def interior_queries(t, nodes: int, rng, count: int = 50) -> np.ndarray:
    """Random query times that have enough nodes on both sides (and are not nodes)."""
    left, right = window_split(nodes)
    lo, hi = t[left - 1], t[len(t) - right]
    q = rng.uniform(lo, hi, size=count)
    return q[np.min(np.abs(q[:, None] - t[None, :]), axis=1) > 1e-3]


def gap_midpoints(t, nodes: int):
    """
    Middle of every computable gap and the node window used for it.

    Returns
    -------
    q       : ndarray (G,)
    windows : list of ndarray (nodes,) - node times for each query
    """
    left, right = window_split(nodes)
    gaps = np.arange(left, len(t) - right + 1)
    q = (t[gaps - 1] + t[gaps]) / 2
    windows = [t[g - left : g + right] for g in gaps]
    return q, windows
