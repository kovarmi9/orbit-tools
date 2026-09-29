from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# Time parsing
# ============================================================

_MJD_EPOCH = datetime(1858, 11, 17)


def _iso_to_seconds(s: str) -> float:
    """Parse ISO datetime string and return seconds since MJD epoch."""
    dt = datetime.fromisoformat(s)
    return (dt - _MJD_EPOCH).total_seconds()


def _mjd_to_seconds(s: str) -> float:
    """Parse MJD float string and return seconds since MJD epoch."""
    return float(s) * 86400.0


def _detect_time_format(first_token: str) -> str:
    """Return 'iso' or 'mjd' based on the first token of the first data line."""
    try:
        float(first_token)
        return "mjd"
    except ValueError:
        return "iso"


# ============================================================
# Reading
# ============================================================

def _iter_data_lines(source):
    """
    Yield non-empty, non-comment lines from a file path or a stream.
    Accepts Path objects, '-' (stdin), or any file-like object.
    """
    if source == "-" or source is sys.stdin:
        raw = sys.stdin.read()
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        raw = Path(source).read_text(encoding="utf-8")
    for ln in raw.splitlines():
        if ln.strip() and not ln.startswith("#"):
            yield ln


def _source_name(source) -> str:
    return "<stdin>" if source == "-" else str(source)


def read_orbit(source, delimiter: str | None):
    """
    Read an sp3_reader output file or stdin ('-').

    Returns
    -------
    t : ndarray (N,)   - seconds since MJD epoch
    r : ndarray (N,3)  - position [m]
    v : ndarray (N,3)  - velocity [m/s], NaN if the line has no velocity columns
    """
    sep = delimiter
    name = _source_name(source)
    lines = list(_iter_data_lines(source))
    if not lines:
        raise ValueError(f"No data found in {name}.")
    fmt     = _detect_time_format(lines[0].split(sep)[0])
    parse_t = _iso_to_seconds if fmt == "iso" else _mjd_to_seconds
    t_list, r_list, v_list = [], [], []
    for ln in lines:
        cols = ln.split(sep)
        if len(cols) < 4:
            print(f"Warning: skipping short line: {ln!r}", file=sys.stderr)
            continue
        try:
            t_i = parse_t(cols[0])
            r_i = [float(cols[1]), float(cols[2]), float(cols[3])]
            v_i = [float(c) for c in cols[4:7]] if len(cols) >= 7 else [float("nan")] * 3
        except (ValueError, IndexError) as exc:
            print(f"Warning: skipping unparseable line ({exc}): {ln!r}", file=sys.stderr)
            continue
        t_list.append(t_i)
        r_list.append(r_i)
        v_list.append(v_i)
    if not t_list:
        raise ValueError(f"No valid records parsed from {name}.")
    return (
        np.array(t_list, dtype=float),
        np.array(r_list, dtype=float),
        np.array(v_list, dtype=float),
    )


def read_times(source, delimiter: str | None) -> tuple[np.ndarray, list[str]]:
    """
    Read only the time column from an sp3_reader output file or stdin ('-').

    Returns
    -------
    t      : ndarray (N,)  - seconds since MJD epoch
    labels : list[str]     - original time strings (preserved for output)
    """
    sep   = delimiter
    name  = _source_name(source)
    lines = list(_iter_data_lines(source))
    if not lines:
        raise ValueError(f"No data found in {name}.")
    fmt     = _detect_time_format(lines[0].split(sep)[0])
    parse_t = _iso_to_seconds if fmt == "iso" else _mjd_to_seconds
    t_list, labels = [], []
    for ln in lines:
        tok = ln.split(sep)[0]
        try:
            t_list.append(parse_t(tok))
            labels.append(tok)
        except ValueError as exc:
            print(f"Warning: skipping unparseable time ({exc}): {tok!r}", file=sys.stderr)
    return np.array(t_list, dtype=float), labels


# ============================================================
# Writing
# ============================================================

def _format_float(value: float) -> str:
    return f"{value:.12g}"


def write_positions(
    r: np.ndarray,
    time_labels: list[str],
    *,
    delimiter: str | None,
    output_file,
    column_headers: bool = False,
) -> int:
    """
    Write interpolated positions; rows containing NaN are skipped.

    Output columns:
    time x_m y_m z_m

    Returns
    -------
    int - number of skipped rows
    """
    columns = ["time", "x_m", "y_m", "z_m"]
    aligned = delimiter is None

    rows = []
    skipped = 0
    for i, t_str in enumerate(time_labels):
        if np.any(np.isnan(r[i])):
            skipped += 1
            continue
        x, y, z = r[i]
        rows.append([t_str, _format_float(x), _format_float(y), _format_float(z)])

    if aligned:
        col_widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))
        if column_headers:
            print(" ".join(c.rjust(col_widths[i]) for i, c in enumerate(columns)), file=output_file)
        for row in rows:
            print(" ".join(val.rjust(col_widths[i]) for i, val in enumerate(row)), file=output_file)
    else:
        if column_headers:
            print(delimiter.join(columns), file=output_file)
        for row in rows:
            print(delimiter.join(row), file=output_file)

    return skipped
