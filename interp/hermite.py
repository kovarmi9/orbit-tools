from __future__ import annotations

import numpy as np

from ._nodes import interpolate_local, prepare
from .registry import register, tied_degree


# ============================================================
# Hermite interpolation  (ported verbatim from
#   MasterThesis-DorisAnalysis/src/doris/analysis/orbits/interpolate/hermite.py)
# ============================================================

def lagrange_basis(x_nodes: np.ndarray, x: float) -> np.ndarray:
    """
    Lagrange basis values L_k(x) for all nodes.

    Parameters
    ----------
    x_nodes : ndarray (m,)
    x : float

    Returns
    -------
    L : ndarray (m,)
    """
    x_nodes = np.asarray(x_nodes, dtype=float).reshape(-1)
    m = x_nodes.size
    L = np.ones(m, dtype=float)

    for k in range(m):
        for j in range(m):
            if j == k:
                continue
            L[k] *= (x - x_nodes[j]) / (x_nodes[k] - x_nodes[j])

    return L


def lagrange_basis_deriv_at_nodes(x_nodes: np.ndarray) -> np.ndarray:
    """
    Derivative of Lagrange basis at its own nodes: L'_k(x_k).

    Parameters
    ----------
    x_nodes : ndarray (m,)

    Returns
    -------
    dL : ndarray (m,)
    """
    x_nodes = np.asarray(x_nodes, dtype=float).reshape(-1)
    m = x_nodes.size
    dL = np.zeros(m, dtype=float)

    for k in range(m):
        s = 0.0
        for j in range(m):
            if j == k:
                continue
            s += 1.0 / (x_nodes[k] - x_nodes[j])
        dL[k] = s

    return dL


def hermite_interpolate(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    dy_nodes: np.ndarray,
    x: float,
) -> np.ndarray:
    """
    Classical Hermite interpolation using values and first derivatives.

    Parameters
    ----------
    x_nodes : ndarray (m,)
    y_nodes : ndarray (m,3)
    dy_nodes : ndarray (m,3)
    x : float

    Returns
    -------
    y : ndarray (3,)
        Interpolated position.
    """
    x_nodes = np.asarray(x_nodes, dtype=float).reshape(-1)
    y_nodes = np.asarray(y_nodes, dtype=float)
    dy_nodes = np.asarray(dy_nodes, dtype=float)

    m = x_nodes.size
    if y_nodes.shape != (m, 3) or dy_nodes.shape != (m, 3):
        raise ValueError("Expected shapes x_nodes(m,), y_nodes(m,3), dy_nodes(m,3).")

    L = lagrange_basis(x_nodes, float(x))
    dL = lagrange_basis_deriv_at_nodes(x_nodes)

    y = np.zeros(3, dtype=float)

    for k in range(m):
        dx = float(x) - x_nodes[k]
        L2 = L[k] * L[k]
        H = (1.0 - 2.0 * dx * dL[k]) * L2
        Hhat = dx * L2
        y += H * y_nodes[k] + Hhat * dy_nodes[k]

    return y


# ============================================================
# Registered method
# ============================================================

@register(
    "hermite",
    summary="positions + velocities, degree = 2n-1 (default: 6 nodes, degree 11)",
    uses_velocity=True,
    resolve=tied_degree(
        default_nodes=6,
        min_nodes=2,
        degree_of=lambda n: 2 * n - 1,
        nodes_of=lambda g: (g + 1) // 2,
        degree_ok=lambda g: g >= 3 and g % 2 == 1,
        degree_rule="an odd number >= 3",
    ),
)
def run(t, r, v, t_query, *, nodes: int, degree: int) -> np.ndarray:
    """Interpolate positions at t_query using Hermite polynomials of degree 2*nodes - 1."""
    t, r, v = prepare(t, r, v, uses_velocity=True)
    return interpolate_local(t, r, v, t_query, n_nodes=nodes, kernel=hermite_interpolate)
