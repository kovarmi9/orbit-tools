"""
Orbit interpolation methods.

Each method lives in its own module and registers itself with @register
(see registry.py). To add a method, create interp/<name>.py and import it
below - an explicit import is required so that PyInstaller bundles it.
"""
from .registry import METHODS, Method, get_method

from . import hermite   # noqa: F401  (registers "hermite")
from . import lagrange  # noqa: F401  (registers "lagrange")
from . import spline    # noqa: F401  (registers "spline")

__all__ = ["METHODS", "Method", "get_method"]
