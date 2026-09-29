"""
Command-line tests of orbit_interpolate.py.

Every variant of input/output (headers, delimiter, stdin, MJD, file vs stdout)
must give the same numbers as the default run; errors must give a readable
message and a non-zero exit code.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "test_ssa_gps.txt"        # CNES orbit, interpolated
REFERENCE = REPO / "test_gop_gps.txt"   # GOP orbit, defines query epochs
EPOCHS = 1441                           # epochs in REFERENCE


def _cli(*args, stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run orbit_interpolate.py; does not raise on non-zero exit."""
    return subprocess.run(
        [sys.executable, str(REPO / "orbit_interpolate.py"), *map(str, args)],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _ok(*args, stdin: str | None = None) -> str:
    """Run orbit_interpolate.py, require success and return stdout."""
    result = _cli(*args, stdin=stdin)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _rows(text: str, delimiter: str | None = None) -> list[list[str]]:
    """Data rows split into columns, without '#' comments."""
    return [
        [col.strip() for col in ln.split(delimiter)]
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def _sp3_reader(sp3: str, out: Path, *args) -> Path:
    subprocess.run(
        [sys.executable, str(REPO / "sp3_reader.py"), str(REPO / sp3), "-t", "GPS", *args, "-o", str(out)],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="module")
def default_rows() -> list[list[str]]:
    """Output of the default run - the baseline all variants are compared to."""
    rows = _rows(_ok(DATA, REFERENCE))
    assert len(rows) == EPOCHS
    return rows


# ============================================================
# Output variants
# ============================================================

def test_output_file_equals_stdout(tmp_path, default_rows):
    out = tmp_path / "int.txt"
    stdout = _ok(DATA, REFERENCE, "-o", out)

    assert stdout == ""
    assert _rows(out.read_text(encoding="utf-8")) == default_rows


def test_settings_comment_is_first_line():
    first = _ok(DATA, REFERENCE, "-g", "7").splitlines()[0]
    assert first == "# orbit_interpolate method=hermite nodes=4 degree=7"


def test_column_headers(default_rows):
    rows = _rows(_ok(DATA, REFERENCE, "-c"))

    assert rows[0] == ["time", "x_m", "y_m", "z_m"]
    assert rows[1:] == default_rows


def test_semicolon_delimiter(tmp_path, default_rows):
    """-d applies to inputs and output: semicolon inputs from sp3_reader -d ';'."""
    data = _sp3_reader("test_ssa.sp3", tmp_path / "ssa.txt", "-d", ";")
    ref = _sp3_reader("test_gop.sp3", tmp_path / "gop.txt", "-d", ";")

    text = _ok(data, ref, "-d", ";", "-c")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]

    assert lines[0] == "time;x_m;y_m;z_m"
    assert all(ln.count(";") == 3 and " " not in ln for ln in lines)
    assert _rows(text, ";")[1:] == default_rows


def test_mjd_times(tmp_path, default_rows):
    """MJD inputs give the same positions; time labels are kept as MJD."""
    data = _sp3_reader("test_ssa.sp3", tmp_path / "ssa.txt", "-f", "mjd")
    ref = _sp3_reader("test_gop.sp3", tmp_path / "gop.txt", "-f", "mjd")

    rows = _rows(_ok(data, ref))

    assert [r[1:] for r in rows] == [r[1:] for r in default_rows]
    assert rows[0][0].startswith("60310")  # 2024-01-01 = MJD 60310


@pytest.mark.parametrize("which", ["data", "reference"])
def test_stdin(which, default_rows):
    if which == "data":
        stdout = _ok("-", REFERENCE, stdin=DATA.read_text(encoding="utf-8"))
    else:
        stdout = _ok(DATA, "-", stdin=REFERENCE.read_text(encoding="utf-8"))

    assert _rows(stdout) == default_rows


def test_nodes_and_degree_are_equivalent():
    assert _ok(DATA, REFERENCE, "-n", "4") == _ok(DATA, REFERENCE, "-g", "7")


def test_default_equals_explicit_hermite_degree_11():
    assert _ok(DATA, REFERENCE) == _ok(DATA, REFERENCE, "-m", "hermite", "-g", "11")


def test_epochs_outside_data_are_skipped_with_warning(tmp_path):
    """Data ending early: late epochs are skipped, reported on stderr, exit code stays 0."""
    short = tmp_path / "short.txt"
    short.write_text("\n".join(DATA.read_text(encoding="utf-8").splitlines()[:300]) + "\n", encoding="utf-8")

    result = _cli(short, REFERENCE)
    assert result.returncode == 0, result.stderr

    match = re.search(r"Warning: (\d+) epoch\(s\) skipped", result.stderr)
    assert match, result.stderr
    skipped = int(match.group(1))

    assert 0 < skipped < EPOCHS
    assert len(_rows(result.stdout)) + skipped == EPOCHS


# ============================================================
# Position-only data
# ============================================================

@pytest.fixture(scope="module")
def position_only_data(tmp_path_factory) -> Path:
    """DATA with the velocity columns removed (time x y z)."""
    out = tmp_path_factory.mktemp("pos") / "ssa_pos.txt"
    lines = [" ".join(ln.split()[:4]) for ln in DATA.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def test_lagrange_accepts_position_only_data(position_only_data):
    """Lagrange ignores velocities, so dropping them does not change the result."""
    full = _ok(DATA, REFERENCE, "-m", "lagrange")
    pos_only = _ok(position_only_data, REFERENCE, "-m", "lagrange")

    assert _rows(pos_only) == _rows(full)
    assert len(_rows(full)) == EPOCHS


def test_lagrange_differs_from_hermite(default_rows):
    """Sanity check that -m really switches the method."""
    assert _rows(_ok(DATA, REFERENCE, "-m", "lagrange")) != default_rows


def test_spline_accepts_position_only_data(position_only_data):
    full = _ok(DATA, REFERENCE, "-m", "spline")
    pos_only = _ok(position_only_data, REFERENCE, "-m", "spline")

    assert _rows(pos_only) == _rows(full)
    assert len(_rows(full)) == EPOCHS
    assert full.splitlines()[0] == "# orbit_interpolate method=spline nodes=all degree=3"


def test_hermite_rejects_position_only_data(position_only_data):
    result = _cli(position_only_data, REFERENCE, "-m", "hermite")

    assert result.returncode == 1
    assert "method 'hermite' needs velocities, but DATA_FILE has none" in result.stderr
    assert "position-only methods: lagrange, spline" in result.stderr
    assert result.stdout == ""


# ============================================================
# Informational options
# ============================================================

def test_list_methods_needs_no_files():
    result = _cli("--list-methods")

    assert result.returncode == 0
    assert re.search(r"^\s*hermite\s+pos\+vel\s", result.stdout, re.MULTILINE)
    assert re.search(r"^\s*lagrange\s+pos\s", result.stdout, re.MULTILINE)
    assert re.search(r"^\s*spline\s+pos\s", result.stdout, re.MULTILINE)


def test_help():
    result = _cli("-h")

    assert result.returncode == 0
    for option in ("--method", "--nodes", "--degree", "--list-methods", "--column-headers", "--delimiter", "--output"):
        assert option in result.stdout


# ============================================================
# Errors
# ============================================================

@pytest.mark.parametrize(
    "args, message",
    [
        (["-g", "8"],             "degree must be an odd number >= 3"),
        (["-n", "1"],             "nodes must be >= 2"),
        (["-n", "4", "-g", "11"], "inconsistent"),
        (["-m", "no-such"],       "unknown method 'no-such' (available: hermite, lagrange, spline)"),
        (["-m", "spline", "-n", "6"], "spline uses all data points; --nodes does not apply"),
        (["-m", "spline", "-g", "5"], "spline degree is fixed at 3"),
        (["-n", "20000"],         "Error during interpolation: Not enough points"),
    ],
)
def test_invalid_settings(args, message):
    result = _cli(DATA, REFERENCE, *args)

    assert result.returncode == 1
    assert message in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "files, message",
    [
        (["missing.txt", REFERENCE], "Error reading DATA_FILE"),
        ([DATA, "missing.txt"],      "Error reading REFERENCE_FILE"),
    ],
)
def test_missing_file(tmp_path, files, message):
    files = [tmp_path / f if f == "missing.txt" else f for f in files]
    result = _cli(*files)

    assert result.returncode == 1
    assert message in result.stderr
    assert result.stdout == ""


def test_empty_data_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n", encoding="utf-8")

    result = _cli(empty, REFERENCE)

    assert result.returncode == 1
    assert "Error reading DATA_FILE: No data found" in result.stderr


def test_both_files_from_stdin_rejected():
    result = _cli("-", "-", stdin="")

    assert result.returncode == 2  # argparse usage error
    assert "cannot both be '-'" in result.stderr


def test_error_does_not_create_output_file(tmp_path):
    out = tmp_path / "int.txt"
    result = _cli(DATA, REFERENCE, "-g", "8", "-o", out)

    assert result.returncode == 1
    assert not out.exists()
