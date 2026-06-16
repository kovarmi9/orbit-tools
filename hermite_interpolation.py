#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# Hermite interpolation  (ported verbatim from
#   MasterThesis-DorisAnalysis/src/doris/analysis/orbits/interpolate/hermite.py)
# ============================================================

def _coerce_inputs(data):
    """
    Supported input formats
    -----------------------
    1) (t, r, v)
       - t: (N,)
       - r: (N,3) or (3,N)
       - v: (N,3) or (3,N)

    2) (t, x, y, z, vx, vy, vz)
       - 7 separate vectors

    3) array (N,7) = [t, x, y, z, vx, vy, vz]

    Returns
    -------
    t : ndarray (N,)
    r : ndarray (N,3)
    v : ndarray (N,3)
    """
    if isinstance(data, (tuple, list)):
        if len(data) == 3:
            t, r, v = data
            t = np.asarray(t, dtype=float).reshape(-1)
            r = np.asarray(r, dtype=float)
            v = np.asarray(v, dtype=float)

            if r.ndim == 2 and r.shape == (3, t.size):
                r = r.T
            if v.ndim == 2 and v.shape == (3, t.size):
                v = v.T

            return t, r, v

        if len(data) == 7:
            t, x, y, z, vx, vy, vz = [np.asarray(a, dtype=float).reshape(-1) for a in data]
            r = np.column_stack([x, y, z])
            v = np.column_stack([vx, vy, vz])
            return t, r, v

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 7:
        t = arr[:, 0]
        r = arr[:, 1:4]
        v = arr[:, 4:7]
        return t, r, v

    raise TypeError(
        "data must be one of: "
        "(t, r, v), (t, x, y, z, vx, vy, vz), "
        "or array (N,7)=[t,x,y,z,vx,vy,vz]."
    )


def _as_query_array(t_query):
    """
    Convert query times to 1D array.

    Returns
    -------
    tq : ndarray (Q,)
    was_scalar : bool
    """
    tq = np.asarray(t_query, dtype=float)
    if tq.ndim == 0:
        return tq.reshape(1), True
    return tq.reshape(-1), False


def _exact_node_index(t_sorted: np.ndarray, q: float, atol: float) -> int | None:
    """
    Return exact node index if q matches t_sorted within tolerance, else None.
    """
    k = int(np.searchsorted(t_sorted, q))

    if k < len(t_sorted) and abs(t_sorted[k] - q) <= atol:
        return k
    if k > 0 and abs(t_sorted[k - 1] - q) <= atol:
        return k - 1

    return None


def _select_nodes(t_sorted: np.ndarray, t_query: float, n_nodes: int) -> np.ndarray:
    """
    Select n_nodes around the interpolation gap given by searchsorted(t, t_query).

    Parameters
    ----------
    t_sorted : ndarray (N,)
        Strictly increasing times.
    t_query : float
        Query time.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    ndarray (n_nodes,)
        Selected node indices.

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


def hermite_at_time(
    data,
    t_query,
    *,
    degree: int = 11,
    drop_nan: bool = True,
    assume_sorted: bool = False,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Interpolate position at one or more query times using Hermite polynomials.

    Parameters
    ----------
    data : array (N,7) or tuple (t, r, v) or tuple (t,x,y,z,vx,vy,vz)
    t_query : float or array-like  - query time(s) in the same units as t
    degree : int  - polynomial degree, must be odd (default 11 -> 6 nodes)
    drop_nan : bool  - drop NaN rows before interpolation
    assume_sorted : bool  - skip time-sorting if data is already sorted
    atol : float  - tolerance for exact node match

    Returns
    -------
    ndarray (3,) if t_query is scalar, else (Q,3)
    """
    if degree % 2 == 0:
        raise ValueError("degree must be odd (3, 5, 7, 9, 11, ...).")
    n_nodes = (degree + 1) // 2
    t, r, v = _coerce_inputs(data)
    if r.shape != (t.size, 3) or v.shape != (t.size, 3):
        raise ValueError("After coercion, shapes must be t(N,), r(N,3), v(N,3).")
    if drop_nan:
        mask = (np.isfinite(t)
                & np.isfinite(r).all(axis=1)
                & np.isfinite(v).all(axis=1))
        t, r, v = t[mask], r[mask], v[mask]
    if t.size < n_nodes:
        raise ValueError(
            f"Not enough points for degree={degree}: "
            f"need {n_nodes} nodes, got {t.size}."
        )
    if not assume_sorted:
        order = np.argsort(t)
        t, r, v = t[order], r[order], v[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Times must be strictly increasing and without duplicates.")
    tq, was_scalar = _as_query_array(t_query)
    r_out = np.full((tq.size, 3), np.nan, dtype=float)
    for i, q in enumerate(tq):
        j_exact = _exact_node_index(t, float(q), float(atol))
        if j_exact is not None:
            r_out[i] = r[j_exact]
            continue
        try:
            idx = _select_nodes(t, float(q), n_nodes)
        except ValueError:
            continue  # out of range - leave NaN, caller decides what to do
        r_out[i] = hermite_interpolate(t[idx], r[idx], v[idx], float(q))
    if was_scalar:
        return r_out[0]
    return r_out


# ============================================================
# File I/O
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


def _iter_data_lines(source, delimiter: str | None):
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


def _parse_orbit_lines(lines, delimiter: str | None, source_name: str = "<input>"):
    sep = delimiter
    lines = list(lines)
    if not lines:
        raise ValueError(f"No data found in {source_name}.")
    fmt     = _detect_time_format(lines[0].split(sep)[0])
    parse_t = _iso_to_seconds if fmt == "iso" else _mjd_to_seconds
    t_list, r_list, v_list = [], [], []
    for ln in lines:
        cols = ln.split(sep)
        if len(cols) < 7:
            print(f"Warning: skipping short line: {ln!r}", file=sys.stderr)
            continue
        try:
            t_list.append(parse_t(cols[0]))
            r_list.append([float(cols[1]), float(cols[2]), float(cols[3])])
            v_list.append([float(cols[4]), float(cols[5]), float(cols[6])])
        except (ValueError, IndexError) as exc:
            print(f"Warning: skipping unparseable line ({exc}): {ln!r}", file=sys.stderr)
    if not t_list:
        raise ValueError(f"No valid records parsed from {source_name}.")
    return (
        np.array(t_list, dtype=float),
        np.array(r_list, dtype=float),
        np.array(v_list, dtype=float),
    )


def _read_orbit_file(source, delimiter: str | None):
    """
    Read an sp3_reader output file or stdin ('-').

    Returns
    -------
    t : ndarray (N,)   - seconds since MJD epoch
    r : ndarray (N,3)  - position [m]
    v : ndarray (N,3)  - velocity [m/s]
    """
    name = "<stdin>" if source == "-" else str(source)
    return _parse_orbit_lines(_iter_data_lines(source, delimiter), delimiter, name)


def _read_times_only(source, delimiter: str | None) -> tuple[np.ndarray, list[str]]:
    """
    Read only the time column from an sp3_reader output file or stdin ('-').

    Returns
    -------
    t      : ndarray (N,)  - seconds since MJD epoch
    labels : list[str]     - original time strings (preserved for output)
    """
    sep     = delimiter
    lines   = list(_iter_data_lines(source, delimiter))
    name    = "<stdin>" if source == "-" else str(source)
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
# Output
# ============================================================

def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _print_results(
    r_interp: np.ndarray,
    time_labels: list[str],
    delimiter: str | None,
    output_file,
    column_headers: bool = False,
) -> None:
    columns = ["time", "x_m", "y_m", "z_m"]
    aligned = delimiter is None

    rows = []
    skipped = 0
    for i, t_str in enumerate(time_labels):
        if np.any(np.isnan(r_interp[i])):
            skipped += 1
            continue
        x, y, z = r_interp[i]
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

    if skipped:
        print(
            f"Warning: {skipped} epoch(s) skipped - outside interpolation range.",
            file=sys.stderr,
        )


# ============================================================
# CLI
# ============================================================

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    Custom help formatter that shows metavar only once (after the long option)
    and aligns help text further right so all options fit on one line.

    Without this class:  -g N, --degree N
    With this class:     -g, --degree N   Hermite polynomial degree.
    """

    def __init__(self, prog):
        # Push help text further right (position 38) and widen the output (100 chars)
        super().__init__(prog, max_help_position=38, width=100)

    def _format_action_invocation(self, action):
        # For positional args and flags without a value (e.g. --metadata), use default formatting
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)

        metavar = action.metavar or ""
        shorts = [o for o in action.option_strings if not o.startswith("--")]
        longs  = [o for o in action.option_strings if o.startswith("--")]

        # Show metavar only next to the long option: -d, --delimiter SEP
        if metavar:
            return ", ".join(shorts + [f"{o} {metavar}" for o in longs])
        return ", ".join(action.option_strings)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermite_interpolation",
        usage="hermite_interpolation [options] DATA_FILE REFERENCE_FILE",
        description="""\
Hermite-interpolate an SP3 trajectory onto a reference time grid.

DATA_FILE      sp3_reader output whose trajectory is interpolated
REFERENCE_FILE sp3_reader output whose time column defines the query epochs""",
        epilog="""\
output columns:
  time  x_m  y_m  z_m

examples:
  hermite_interpolation data.txt reference.txt
  hermite_interpolation data.txt reference.txt -g 7
  hermite_interpolation data.txt reference.txt -c              print column headers
  hermite_interpolation data.txt reference.txt -d ";"          semicolon-separated output
  hermite_interpolation data.txt reference.txt -o result.txt   write to file""",
        formatter_class=_HelpFormatter,
    )

    parser.add_argument(
        "data_file",
        metavar="DATA_FILE",
        help="Trajectory to interpolate (sp3_reader output).",
    )

    parser.add_argument(
        "reference_file",
        metavar="REFERENCE_FILE",
        help="Reference file whose time column defines the query epochs.",
    )

    parser.add_argument(
        "-g",
        "--degree",
        type=int,
        default=11,
        metavar="N",
        help="Hermite polynomial degree, must be odd (default: 11 -> 6 nodes).",
    )

    parser.add_argument(
        "-c",
        "--column-headers",
        action="store_true",
        help="Print column names as first line of output.",
    )

    parser.add_argument(
        "-d",
        "--delimiter",
        default=None,
        metavar="SEP",
        help="Column separator for inputs and output (default: aligned columns).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # --- stdin can only be consumed once ---
    if args.data_file == "-" and args.reference_file == "-":
        parser.error("DATA_FILE and REFERENCE_FILE cannot both be '-' (stdin).")

    # --- validate degree ---
    if args.degree % 2 == 0 or args.degree < 3:
        print(
            f"Error: --degree must be an odd number >= 3 (got {args.degree}).",
            file=sys.stderr,
        )
        return 1

    # --- read data trajectory ---
    try:
        t_data, r_data, v_data = _read_orbit_file(args.data_file, args.delimiter)
    except (OSError, ValueError) as exc:
        print(f"Error reading DATA_FILE: {exc}", file=sys.stderr)
        return 1

    # --- read reference times ---
    ref_source = "-" if args.reference_file == "-" else Path(args.reference_file)
    try:
        t_query, time_labels = _read_times_only(ref_source, args.delimiter)
    except (OSError, ValueError) as exc:
        print(f"Error reading REFERENCE_FILE: {exc}", file=sys.stderr)
        return 1

    n_nodes = (args.degree + 1) // 2
    print(
        f"Interpolating {len(t_query)} epochs  |  "
        f"degree={args.degree}  nodes={n_nodes}  |  "
        f"data points={len(t_data)}",
        file=sys.stderr,
    )

    # --- interpolate ---
    try:
        r_interp = hermite_at_time(
            (t_data, r_data, v_data),
            t_query,
            degree=args.degree,
        )
    except ValueError as exc:
        print(f"Error during interpolation: {exc}", file=sys.stderr)
        return 1

    # --- output ---
    out_file    = sys.stdout
    close_out   = False

    if args.output is not None:
        out_file  = args.output.open("w", encoding="utf-8", newline="")
        close_out = True

    try:
        _print_results(
            r_interp,
            time_labels,
            delimiter=args.delimiter,
            output_file=out_file,
            column_headers=args.column_headers,
        )
    finally:
        if close_out:
            out_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
