from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ============================================================
# Method registry
# ============================================================

# resolve(nodes, degree) -> (nodes, degree); either input may be None (= use default)
Resolver = Callable[[int | None, int | None], tuple[int, int]]

# run(t, r, v, t_query, *, nodes, degree) -> ndarray (Q, 3), NaN where not computed
Runner = Callable[..., np.ndarray]


@dataclass(frozen=True)
class Method:
    name:          str
    summary:       str
    uses_velocity: bool
    resolve:       Resolver
    run:           Runner


METHODS: dict[str, Method] = {}


def register(name: str, *, summary: str, uses_velocity: bool, resolve: Resolver):
    """
    Decorator registering an interpolation method under the given name.

    The decorated function is the method's runner:
        run(t, r, v, t_query, *, nodes, degree) -> ndarray (Q, 3)
    """
    def decorator(run: Runner) -> Runner:
        if name in METHODS:
            raise RuntimeError(f"Interpolation method {name!r} is registered twice.")
        METHODS[name] = Method(name, summary, uses_velocity, resolve, run)
        return run
    return decorator


def get_method(name: str) -> Method:
    """Return a registered method by name (case-insensitive)."""
    try:
        return METHODS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown method {name!r} (available: {', '.join(METHODS)})"
        ) from None


# ============================================================
# Node / degree resolvers
# ============================================================

def tied_degree(
    *,
    default_nodes: int,
    min_nodes:     int,
    degree_of:     Callable[[int], int],
    nodes_of:      Callable[[int], int],
    degree_ok:     Callable[[int], bool],
    degree_rule:   str,
) -> Resolver:
    """
    Resolver for interpolating polynomials, where the degree follows
    from the number of nodes (e.g. Hermite: degree = 2n - 1).

    The user may give nodes, degree, or both (they must agree).
    """
    def resolve(nodes: int | None, degree: int | None) -> tuple[int, int]:
        if degree is not None:
            if not degree_ok(degree):
                raise ValueError(f"degree must be {degree_rule} (got {degree})")
            implied = nodes_of(degree)
            if nodes is not None and nodes != implied:
                raise ValueError(
                    f"nodes={nodes} and degree={degree} are inconsistent "
                    f"(degree {degree} means {implied} nodes)"
                )
            nodes = implied

        if nodes is None:
            nodes = default_nodes

        if nodes < min_nodes:
            raise ValueError(f"nodes must be >= {min_nodes} (got {nodes})")

        return nodes, degree_of(nodes)

    return resolve
