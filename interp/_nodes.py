from __future__ import annotations

from typing import Callable

import numpy as np


# ============================================================
# Data preparation
# ============================================================

def prepare(
    t,
    r,
    v,
    *,
    uses_velocity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate shapes, drop NaN rows and sort by time.

    Velocities are checked for NaN only if the method uses them.

    Returns
    -------
    t : ndarray (N,)
    r : ndarray (N,3)
    v : ndarray (N,3)
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    if r.shape != (t.size, 3) or v.shape != (t.size, 3):
        raise ValueError("Expected shapes t(N,), r(N,3), v(N,3).")

    mask = np.isfinite(t) & np.isfinite(r).all(axis=1)
    if uses_velocity:
        mask &= np.isfinite(v).all(axis=1)
    t, r, v = t[mask], r[mask], v[mask]

    order = np.argsort(t)
    t, r, v = t[order], r[order], v[order]

    if np.any(np.diff(t) <= 0):
        raise ValueError("Times must be strictly increasing and without duplicates.")

    return t, r, v


# ============================================================
# Node selection
# ============================================================

def exact_node_index(t_sorted: np.ndarray, q: float, atol: float) -> int | None:
    """
    Return exact node index if q matches t_sorted within tolerance, else None.
    """
    k = int(np.searchsorted(t_sorted, q))

    if k < len(t_sorted) and abs(t_sorted[k] - q) <= atol:
        return k
    if k > 0 and abs(t_sorted[k - 1] - q) <= atol:
        return k - 1

    return None


def select_nodes(t_sorted: np.ndarray, t_query: float, n_nodes: int) -> np.ndarray:
    """
    Select n_nodes around the interpolation gap given by searchsorted(t, t_query).

    The query always lies in the central gap of the window
    (n_nodes // 2 nodes on the left, the rest on the right).

    Raises
    ------
    ValueError
        If query is too close to edge and enough nodes cannot be selected.
    """
    left_count = n_nodes // 2
    right_count = n_nodes - left_count

    i_gap = int(np.searchsorted(t_sorted, float(t_query)))

    if i_gap - left_count < 0 or i_gap + right_count > len(t_sorted):
        raise ValueError(
            "Query too close to edge for selected degree; not enough nodes on both sides."
        )

    return np.arange(i_gap - left_count, i_gap + right_count)


# ============================================================
# Sliding-window driver
# ============================================================

Kernel = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


def interpolate_local(
    t: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    t_query,
    *,
    n_nodes: int,
    kernel: Kernel,
    exact_nodes: bool = True,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Evaluate a local method at each query time using a sliding window of nodes.

    Parameters
    ----------
    t, r, v : prepared data (see prepare)
    t_query : float or array-like - query time(s) in the same units as t
    n_nodes : int - number of nodes in the window
    kernel : kernel(t_nodes, r_nodes, v_nodes, q) -> ndarray (3,)
    exact_nodes : bool - return the node itself if the query matches it
        (correct for interpolation; must be False for fitting methods)
    atol : float - tolerance for exact node match

    Returns
    -------
    ndarray (Q,3) - NaN for queries outside the interpolation range
    """
    if t.size < n_nodes:
        raise ValueError(f"Not enough points: need {n_nodes} nodes, got {t.size}.")

    tq = np.asarray(t_query, dtype=float).reshape(-1)
    r_out = np.full((tq.size, 3), np.nan, dtype=float)

    for i, q in enumerate(tq):
        if exact_nodes:
            j_exact = exact_node_index(t, float(q), float(atol))
            if j_exact is not None:
                r_out[i] = r[j_exact]
                continue
        try:
            idx = select_nodes(t, float(q), n_nodes)
        except ValueError:
            continue  # out of range - leave NaN, caller decides what to do
        r_out[i] = kernel(t[idx], r[idx], v[idx], float(q))

    return r_out
