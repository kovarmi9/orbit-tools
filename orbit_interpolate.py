#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from interp import METHODS, get_method
from interp.io import read_orbit, read_times, write_positions


# ============================================================
# CLI
# ============================================================

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    Custom help formatter that shows metavar only once (after the long option)
    and aligns help text further right so all options fit on one line.

    Without this class:  -m NAME, --method NAME
    With this class:     -m, --method NAME   Interpolation method.
    """

    def __init__(self, prog):
        # Push help text further right (position 38) and widen the output (100 chars)
        super().__init__(prog, max_help_position=38, width=100)

    def _format_action_invocation(self, action):
        # For positional args and flags without a value (e.g. --list-methods), use default formatting
        if not action.option_strings or action.nargs == 0:
            return super()._format_action_invocation(action)

        metavar = action.metavar or ""
        shorts = [o for o in action.option_strings if not o.startswith("--")]
        longs  = [o for o in action.option_strings if o.startswith("--")]

        # Show metavar only next to the long option: -d, --delimiter SEP
        if metavar:
            return ", ".join(shorts + [f"{o} {metavar}" for o in longs])
        return ", ".join(action.option_strings)


class _ListMethodsAction(argparse.Action):
    """Print available methods and exit (like --help, positional args are not required)."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        for method in METHODS.values():
            inputs = "pos+vel" if method.uses_velocity else "pos"
            print(f"  {method.name:<10} {inputs:<8} {method.summary}")
        parser.exit()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbit_interpolate",
        usage="orbit_interpolate [options] DATA_FILE REFERENCE_FILE",
        description="""\
Interpolate an orbit onto the time grid of a reference orbit.

DATA_FILE      sp3_reader output whose trajectory is interpolated
REFERENCE_FILE sp3_reader output whose time column defines the query epochs""",
        epilog="""\
output columns:
  time  x_m  y_m  z_m
  (preceded by a '# orbit_interpolate ...' comment line with the settings used)

notes:
  both files must use the same time scale (sp3_reader -t)
  --nodes and --degree are two ways to set the same thing; give either one

examples:
  orbit_interpolate data.txt reference.txt
  orbit_interpolate data.txt reference.txt -m hermite -g 7     degree 7 (4 nodes)
  orbit_interpolate data.txt reference.txt -n 8                8 nodes
  orbit_interpolate data.txt reference.txt -c                  print column headers
  orbit_interpolate data.txt reference.txt -d ";"              semicolon-separated output
  orbit_interpolate data.txt reference.txt -o result.txt       write to file
  orbit_interpolate --list-methods""",
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
        "-m",
        "--method",
        default="hermite",
        metavar="NAME",
        help="Interpolation method (default: hermite).",
    )

    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        metavar="N",
        help="Number of nodes around each epoch (default: per method).",
    )

    parser.add_argument(
        "-g",
        "--degree",
        type=int,
        metavar="N",
        help="Polynomial degree, alternative to --nodes (default: per method).",
    )

    parser.add_argument(
        "--list-methods",
        action=_ListMethodsAction,
        help="List available methods and exit.",
    )

    parser.add_argument(
        "-c",
        "--column-headers",
        action="store_true",
        help="Print column names as first data line of output.",
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


# ============================================================
# Main program
# ============================================================

def main() -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    # --- stdin can only be consumed once ---
    if args.data_file == "-" and args.reference_file == "-":
        parser.error("DATA_FILE and REFERENCE_FILE cannot both be '-' (stdin).")

    # --- method and its settings ---
    try:
        method = get_method(args.method)
        nodes, degree = method.resolve(args.nodes, args.degree)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    settings = f"method={method.name} nodes={nodes} degree={degree}"

    # --- read data trajectory ---
    try:
        t_data, r_data, v_data = read_orbit(args.data_file, args.delimiter)
    except (OSError, ValueError) as exc:
        print(f"Error reading DATA_FILE: {exc}", file=sys.stderr)
        return 1

    # --- read reference times ---
    ref_source = "-" if args.reference_file == "-" else Path(args.reference_file)
    try:
        t_query, time_labels = read_times(ref_source, args.delimiter)
    except (OSError, ValueError) as exc:
        print(f"Error reading REFERENCE_FILE: {exc}", file=sys.stderr)
        return 1

    print(
        f"Interpolating {len(t_query)} epochs  |  {settings}  |  data points={len(t_data)}",
        file=sys.stderr,
    )

    # --- interpolate ---
    try:
        r_interp = method.run(t_data, r_data, v_data, t_query, nodes=nodes, degree=degree)
    except ValueError as exc:
        print(f"Error during interpolation: {exc}", file=sys.stderr)
        return 1

    # --- output ---
    out_file  = sys.stdout
    close_out = False

    if args.output is not None:
        out_file  = args.output.open("w", encoding="utf-8", newline="")
        close_out = True

    try:
        # '#' lines are skipped by orbit_interpolate and orbit_diff readers
        print(f"# orbit_interpolate {settings}", file=out_file)
        skipped = write_positions(
            r_interp,
            time_labels,
            delimiter=args.delimiter,
            output_file=out_file,
            column_headers=args.column_headers,
        )
    finally:
        if close_out:
            out_file.close()

    if skipped:
        print(
            f"Warning: {skipped} epoch(s) skipped - outside interpolation range.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
