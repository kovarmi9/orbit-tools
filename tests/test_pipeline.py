"""
End-to-end tests on the test data in the repository.

- regression: the whole chain must reproduce diff.txt byte for byte
- accuracy: Hermite error on real data (thinning test, as in Zeitlhöfler et al. 2024)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from interp import get_method
from interp.io import read_orbit

REPO = Path(__file__).resolve().parent.parent


def _run_tool(script: str, *args) -> None:
    subprocess.run(
        [sys.executable, str(REPO / script), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )


def _data_lines(path: Path) -> list[str]:
    """Data lines without '#' comments, independent of line endings."""
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]


# ============================================================
# Regression
# ============================================================

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    out = tmp_path_factory.mktemp("pipeline")
    gop, ssa, ssa_int, diff = (out / n for n in ("gop.txt", "ssa.txt", "ssa_int.txt", "diff.txt"))

    _run_tool("sp3_reader.py", REPO / "test_gop.sp3", "-t", "GPS", "-o", gop)
    _run_tool("sp3_reader.py", REPO / "test_ssa.sp3", "-t", "GPS", "-o", ssa)
    _run_tool("orbit_interpolate.py", ssa, gop, "-m", "hermite", "-o", ssa_int)
    _run_tool("orbit_diff.py", gop, ssa_int, "-r", "-c", "-o", diff)

    return {"gop": gop, "ssa": ssa, "ssa_int": ssa_int, "diff": diff}


def test_interpolated_orbit_matches_reference(pipeline):
    assert _data_lines(pipeline["ssa_int"]) == _data_lines(REPO / "test_ssa_int.txt")


def test_interpolated_orbit_records_settings(pipeline):
    first = pipeline["ssa_int"].read_text(encoding="utf-8").splitlines()[0]
    assert first == "# orbit_interpolate method=hermite nodes=6 degree=11"


def test_diff_matches_reference_byte_for_byte(pipeline):
    assert pipeline["diff"].read_bytes() == (REPO / "diff.txt").read_bytes()


# ============================================================
# Accuracy on real data
# ============================================================

@pytest.mark.parametrize("nodes", [4, 6])
def test_hermite_accuracy_on_real_orbit(nodes):
    """
    Keep every 2nd epoch of the CNES orbit (60 s -> 120 s) and predict the others.

    The error floor (~0.6-0.7 mm) is given by the 1 mm rounding of SP3 positions.
    Larger outliers (up to ~16 mm) come from discontinuities between daily arcs
    in the CNES file, so only RMS and median are checked.
    """
    t, r, v = read_orbit(REPO / "test_ssa_gps.txt", None)
    keep = np.zeros(t.size, dtype=bool)
    keep[::2] = True

    method = get_method("hermite")
    _, degree = method.resolve(nodes, None)
    pred = method.run(t[keep], r[keep], v[keep], t[~keep], nodes=nodes, degree=degree)

    err_mm = np.linalg.norm(pred - r[~keep], axis=1) * 1000.0
    err_mm = err_mm[np.isfinite(err_mm)]

    assert err_mm.size > 5000
    assert np.sqrt(np.mean(err_mm**2)) < 1.0
    assert np.median(err_mm) < 1.0
