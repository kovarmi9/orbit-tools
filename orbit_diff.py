#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ============================================================
# RTN frame
# ============================================================

_EPS = 1e-15


def _build_rtn_frame(
    position: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_norm = np.linalg.norm(position, axis=1, keepdims=True)
    R_hat  = position / np.maximum(r_norm, _EPS)

    h      = np.cross(position, velocity)
    h_norm = np.linalg.norm(h, axis=1, keepdims=True)
    N_hat  = h / np.maximum(h_norm, _EPS)

    T_hat  = np.cross(N_hat, R_hat)
    return R_hat, T_hat, N_hat


def _project_to_rtn(
    diff_xyz: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    R_hat, T_hat, N_hat = _build_rtn_frame(position, velocity)
    dR = np.sum(diff_xyz * R_hat, axis=1)
    dT = np.sum(diff_xyz * T_hat, axis=1)
    dN = np.sum(diff_xyz * N_hat, axis=1)
    return np.column_stack([dR, dT, dN])


# ============================================================
# File I/O
# ============================================================

def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _iter_data_lines(source, delimiter: str | None):
    if source == "-" or source is sys.stdin:
        raw = sys.stdin.read()
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        raw = Path(source).read_text(encoding="utf-8")
    for ln in raw.splitlines():
        if ln.strip() and not ln.startswith("#"):
            yield ln


def _read_orbit_file(
    source, delimiter: str | None
) -> tuple[list[str], np.ndarray, np.ndarray, bool]:
    """
    Read a 7-column sp3_reader output file.

    Returns
    -------
    labels      : list[str]     - original time strings
    r           : ndarray (N,3) - position [m]
    v           : ndarray (N,3) - velocity [m/s], NaN if missing
    has_velocity: bool          - True if all records have valid velocity
    """
    sep  = delimiter
    name = "<stdin>" if source == "-" else str(source)
    lines = list(_iter_data_lines(source, delimiter))
    if not lines:
        raise ValueError(f"No data found in {name}.")

    # skip header if first token is not numeric/ISO
    first_tok = lines[0].split(sep)[0]
    is_numeric = False
    try:
        float(first_tok)
        is_numeric = True
    except ValueError:
        pass
    if not is_numeric:
        try:
            from datetime import datetime
            datetime.fromisoformat(first_tok)
            is_numeric = True
        except ValueError:
            pass
    if not is_numeric:
        print(f"Warning: skipping header line: {lines[0]!r}", file=sys.stderr)
        lines = lines[1:]

    if not lines:
        raise ValueError(f"No data found in {name} after skipping header.")

    labels, r_list, v_list = [], [], []
    for ln in lines:
        cols = ln.split(sep)
        if len(cols) < 4:
            print(f"Warning: skipping short line: {ln!r}", file=sys.stderr)
            continue
        try:
            r = [float(cols[1]), float(cols[2]), float(cols[3])]
            if len(cols) >= 7:
                v = [float(cols[4]), float(cols[5]), float(cols[6])]
            else:
                v = [float("nan")] * 3
            labels.append(cols[0])
            r_list.append(r)
            v_list.append(v)
        except (ValueError, IndexError) as exc:
            print(f"Warning: skipping unparseable line ({exc}): {ln!r}", file=sys.stderr)

    if not labels:
        raise ValueError(f"No valid records parsed from {name}.")

    r = np.array(r_list, dtype=float)
    v = np.array(v_list, dtype=float)
    has_velocity = bool(np.isfinite(v).all())
    return labels, r, v, has_velocity


# ============================================================
# Output
# ============================================================

def _print_results(
    time_labels: list[str],
    diff_pos: np.ndarray,
    diff_vel: np.ndarray | None,
    pos_cols: list[str],
    vel_cols: list[str] | None,
    delimiter: str | None,
    output_file,
    column_headers: bool = False,
) -> None:
    columns = ["time"] + pos_cols
    if vel_cols is not None:
        columns += vel_cols

    aligned = delimiter is None

    rows = []
    for i, t_str in enumerate(time_labels):
        row = [t_str] + [_format_float(v) for v in diff_pos[i]]
        if diff_vel is not None:
            row += [_format_float(v) for v in diff_vel[i]]
        rows.append(row)

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


# ============================================================
# CLI
# ============================================================

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    Custom help formatter that shows metavar only once (after the long option)
    and aligns help text further right so all options fit on one line.

    Without this class:  -d SEP, --delimiter SEP
    With this class:     -d, --delimiter SEP   Column separator.
    """

    def __init__(self, prog):
        super().__init__(prog, max_help_position=38, width=100)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)
        metavar = action.metavar or ""
        shorts = [o for o in action.option_strings if not o.startswith("--")]
        longs  = [o for o in action.option_strings if o.startswith("--")]
        if metavar:
            return ", ".join(shorts + [f"{o} {metavar}" for o in longs])
        return ", ".join(action.option_strings)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbit_diff",
        usage="orbit_diff [options] FILE_A FILE_B",
        description="""\
Compute epoch-by-epoch position differences between two orbit files.

FILE_A and FILE_B must be sp3_reader output files with matching time columns.
Differences are computed as FILE_B − FILE_A.
Velocity differences are included automatically if both files contain velocities.""",
        epilog="""\
output columns (XYZ):   time  dx_m  dy_m  dz_m  [dvx_mps  dvy_mps  dvz_mps]
output columns (RTN):   time  dR_m  dT_m  dN_m  [dvR_mps  dvT_mps  dvN_mps]

notes:
  velocity columns are included only when both files contain valid velocities
  norm_m is always the 3D norm of the position difference
  RTN frame is built from FILE_A position and velocity
  epochs are matched by time string — use hermite_interpolation to align first

examples:
  orbit_diff file_a.txt file_b.txt
  orbit_diff file_a.txt file_b.txt -r
  orbit_diff file_a.txt file_b.txt -c              print column headers
  orbit_diff file_a.txt file_b.txt -d ";"          semicolon-separated output
  orbit_diff file_a.txt file_b.txt -o diff.txt     write to file""",
        formatter_class=_HelpFormatter,
    )

    parser.add_argument(
        "file_a",
        metavar="FILE_A",
        help="Reference orbit file (sp3_reader output).",
    )

    parser.add_argument(
        "file_b",
        metavar="FILE_B",
        help="Orbit file to compare against FILE_A (sp3_reader output).",
    )

    parser.add_argument(
        "-r",
        "--rtn",
        action="store_true",
        help="Output differences in RTN frame instead of XYZ.",
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

    if args.file_a == "-" and args.file_b == "-":
        parser.error("FILE_A and FILE_B cannot both be '-' (stdin).")

    # --- read both files ---
    try:
        labels_a, r_a, v_a, has_vel_a = _read_orbit_file(args.file_a, args.delimiter)
    except (OSError, ValueError) as exc:
        print(f"Error reading FILE_A: {exc}", file=sys.stderr)
        return 1

    try:
        labels_b, r_b, v_b, has_vel_b = _read_orbit_file(args.file_b, args.delimiter)
    except (OSError, ValueError) as exc:
        print(f"Error reading FILE_B: {exc}", file=sys.stderr)
        return 1

    has_velocity = has_vel_a and has_vel_b
    if has_vel_a != has_vel_b:
        print(
            "Warning: one file is missing velocities — velocity differences will not be computed.",
            file=sys.stderr,
        )

    # --- match epochs by time string ---
    index_a = {t: i for i, t in enumerate(labels_a)}
    matched_labels, idx_a, idx_b = [], [], []
    skipped = 0
    for j, t in enumerate(labels_b):
        if t in index_a:
            matched_labels.append(t)
            idx_a.append(index_a[t])
            idx_b.append(j)
        else:
            skipped += 1

    if skipped:
        print(
            f"Warning: {skipped} epoch(s) in FILE_B have no matching epoch in FILE_A and were skipped.",
            file=sys.stderr,
        )

    if not matched_labels:
        print("Error: no matching epochs found between FILE_A and FILE_B.", file=sys.stderr)
        return 1

    r_a_m = r_a[idx_a]
    r_b_m = r_b[idx_b]
    v_a_m = v_a[idx_a]
    v_b_m = v_b[idx_b]

    print(
        f"Comparing {len(matched_labels)} epochs  |  "
        f"{'RTN' if args.rtn else 'XYZ'}  |  "
        f"velocities: {'yes' if has_velocity else 'no'}",
        file=sys.stderr,
    )

    # --- compute differences B - A ---
    diff_xyz     = r_b_m - r_a_m
    diff_vel_xyz = (v_b_m - v_a_m) if has_velocity else None

    if args.rtn:
        if not has_vel_a:
            print("Error: RTN frame requires velocities in FILE_A.", file=sys.stderr)
            return 1
        diff_pos  = _project_to_rtn(diff_xyz, r_a_m, v_a_m)
        diff_vel  = _project_to_rtn(diff_vel_xyz, r_a_m, v_a_m) if has_velocity else None
        pos_cols  = ["dR_m", "dT_m", "dN_m"]
        vel_cols  = ["dvR_mps", "dvT_mps", "dvN_mps"]
    else:
        diff_pos  = diff_xyz
        diff_vel  = diff_vel_xyz
        pos_cols  = ["dx_m", "dy_m", "dz_m"]
        vel_cols  = ["dvx_mps", "dvy_mps", "dvz_mps"] if has_velocity else None

    # --- output ---
    out_file  = sys.stdout
    close_out = False

    if args.output is not None:
        out_file  = args.output.open("w", encoding="utf-8", newline="")
        close_out = True

    try:
        _print_results(
            matched_labels,
            diff_pos,
            diff_vel,
            pos_cols,
            vel_cols if has_velocity else None,
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
