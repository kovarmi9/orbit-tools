#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# File I/O
# ============================================================

def _parse_time(s: str) -> float:
    try:
        return float(s) * 86400.0
    except ValueError:
        return datetime.fromisoformat(s).timestamp()


def _read_diff_file(path: Path, delimiter: str | None):
    """
    Read orbit_diff output file.

    Returns
    -------
    t_hours : ndarray (N,)   - time in hours from first epoch
    data    : ndarray (N, 3) - first three difference columns [m]
    labels  : list[str]      - subplot labels (detected from header or generic)
    """
    sep   = delimiter
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]

    if not lines:
        raise ValueError(f"No data in {path}.")

    # detect header line
    col_names = None
    first_tok = lines[0].split(sep)[0]
    is_header = False
    try:
        float(first_tok)
    except ValueError:
        try:
            datetime.fromisoformat(first_tok)
        except ValueError:
            is_header = True

    if is_header:
        col_names = lines[0].split(sep)
        lines = lines[1:]

    if not lines:
        raise ValueError(f"No data in {path} after skipping header.")

    t_list, d_list = [], []
    for ln in lines:
        cols = ln.split(sep)
        if len(cols) < 4:
            continue
        try:
            t_list.append(_parse_time(cols[0]))
            d_list.append([float(cols[1]), float(cols[2]), float(cols[3])])
        except ValueError:
            continue

    if not t_list:
        raise ValueError(f"No valid records parsed from {path}.")

    t       = np.array(t_list, dtype=float)
    t_hours = (t - t[0]) / 3600.0
    data    = np.array(d_list, dtype=float)

    # detect RTN vs XYZ from header column names
    if col_names is not None:
        names_lower = [c.lower() for c in col_names]
        if any("dr" in n for n in names_lower):
            labels = ["Radiální složka (R)", "Transverzální složka (T)", "Normálová složka (N)"]
        elif any("dx" in n for n in names_lower):
            labels = ["Složka X", "Složka Y", "Složka Z"]
        else:
            labels = ["Složka 1", "Složka 2", "Složka 3"]
    else:
        labels = ["Složka 1", "Složka 2", "Složka 3"]

    return t_hours, data, labels


# ============================================================
# Plot
# ============================================================

def _plot(t_hours: np.ndarray, data: np.ndarray, labels: list[str], output: Path | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Rozdíly orbit", fontsize=14, fontweight="bold")

    data_mm = data * 1000.0  # m → mm

    for ax, col, label in zip(axes, data_mm.T, labels):
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(t_hours, col, color="#1f77b4", linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Rozdíl [mm]", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.7)

    axes[-1].set_xlabel("Čas od začátku série [h]", fontsize=9)

    plt.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}", file=sys.stderr)
    else:
        plt.show()


# ============================================================
# CLI
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_orbit_diff",
        description="Plot orbit_diff output as a 3-panel difference chart (RTN or XYZ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        metavar="FILE",
        help="orbit_diff output file (default: diff.txt next to this script).",
    )

    parser.add_argument(
        "-d",
        "--delimiter",
        default=None,
        metavar="SEP",
        help="Column separator used in input file (default: whitespace).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save plot to FILE instead of showing interactively (e.g. diff.png).",
    )

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args   = parser.parse_args()

    path = Path(args.input) if args.input is not None else Path(__file__).parent / "diff.txt"

    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        t_hours, data, labels = _read_diff_file(path, args.delimiter)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _plot(t_hours, data, labels, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
