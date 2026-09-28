"""
Orbit interpolation methods.

Each method lives in its own module and registers itself with @register
(see registry.py). To add a method, create interp/<name>.py and import it
below - an explicit import is required so that PyInstaller bundles it.
"""
from .registry import METHODS, Method, get_method

from . import hermite  # noqa: F401  (registers "hermite")

__all__ = ["METHODS", "Method", "get_method"]
