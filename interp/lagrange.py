from __future__ import annotations

import numpy as np

from ._nodes import interpolate_local, prepare
from .hermite import lagrange_basis
from .registry import register, tied_degree


# ============================================================
# Lagrange interpolation
# ============================================================

def lagrange_interpolate(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    dy_nodes: np.ndarray,
    x: float,
) -> np.ndarray:
    """
    Classical Lagrange interpolation using values only.

    Parameters
    ----------
    x_nodes : ndarray (m,)
    y_nodes : ndarray (m,3)
    dy_nodes : ndarray (m,3) - ignored (kernel signature shared with Hermite)
    x : float

    Returns
    -------
    y : ndarray (3,)
        Interpolated position.
    """
    y_nodes = np.asarray(y_nodes, dtype=float)
    L = lagrange_basis(x_nodes, float(x))
    return L @ y_nodes


# ============================================================
# Registered method
# ============================================================

@register(
    "lagrange",
    summary="positions only, degree = n-1 (default: 10 nodes, degree 9)",
    uses_velocity=False,
    resolve=tied_degree(
        default_nodes=10,
        min_nodes=2,
        degree_of=lambda n: n - 1,
        nodes_of=lambda g: g + 1,
        degree_ok=lambda g: g >= 1,
        degree_rule=">= 1",
    ),
)
def run(t, r, v, t_query, *, nodes: int, degree: int) -> np.ndarray:
    """Interpolate positions at t_query using Lagrange polynomials of degree nodes - 1."""
    t, r, v = prepare(t, r, v, uses_velocity=False)
    return interpolate_local(t, r, v, t_query, n_nodes=nodes, kernel=lagrange_interpolate)
