#!/usr/bin/env python3
"""
Plot X, Y, Z position differences between two sp3_reader output files.

Usage:
    python plot_orbit_diff.py SSA_FILE GOP_TAI_FILE
    python plot_orbit_diff.py test_ssa.txt test_gop_tai.txt -o diff.pdf
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# I/O
# ============================================================

_MJD_EPOCH = datetime(1858, 11, 17)


def _parse_time(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return _MJD_EPOCH + timedelta(days=float(s))


def _read_file(path: Path) -> tuple[list[datetime], np.ndarray]:
    times, rows = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        cols = ln.split()
        if len(cols) < 4:
            continue
        try:
            times.append(_parse_time(cols[0]))
            rows.append([float(cols[1]), float(cols[2]), float(cols[3])])
        except ValueError:
            continue
    return times, np.array(rows, dtype=float)


# ============================================================
# Plot
# ============================================================

def _plot(
    times: list[datetime],
    diff: np.ndarray,
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    components = ["X", "Y", "Z"]
    units      = "m"

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    fig.suptitle(
        f"Orbit difference  {label_a} - {label_b}",
        fontsize=15,
        fontweight="bold",
    )

    for i, (ax, comp) in enumerate(zip(axes, components)):
        ax.plot(times, diff[:, i], linewidth=0.8, color=f"C{i}")
        ax.axhline(0, color="black", linewidth=1.0, zorder=0)
        ax.set_ylabel(f"$\\Delta {comp}$  [{units}]")
        ax.grid(alpha=0.3)

        rms  = np.sqrt(np.mean(diff[:, i] ** 2))
        mean = np.mean(diff[:, i])
        ax.set_title(
            f"{comp}   mean = {mean:+.4f} m    RMS = {rms:.4f} m",
            fontsize=10,
        )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d. %m. %Y %H:%M"))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}", file=sys.stderr)


# ============================================================
# CLI
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_orbit_diff",
        description="Plot X/Y/Z differences between two sp3_reader output files.",
    )
    p.add_argument("ssa_file",    type=Path, metavar="SSA_FILE")
    p.add_argument("gop_tai_file", type=Path, metavar="GOP_TAI_FILE")
    p.add_argument(
        "-o", "--output", type=Path, metavar="FILE",
        default=Path("orbit_diff.pdf"),
        help="Output PDF (default: orbit_diff.pdf).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()

    try:
        times_a, xyz_a = _read_file(args.ssa_file)
        times_b, xyz_b = _read_file(args.gop_tai_file)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # match on common times
    set_b = {t: i for i, t in enumerate(times_b)}
    common_times, idx_a, idx_b = [], [], []
    for i, t in enumerate(times_a):
        if t in set_b:
            common_times.append(t)
            idx_a.append(i)
            idx_b.append(set_b[t])

    if not common_times:
        print("Error: no common epochs found between the two files.", file=sys.stderr)
        return 1

    diff = xyz_a[idx_a] - xyz_b[idx_b]

    print(
        f"Common epochs: {len(common_times)}  "
        f"({common_times[0].isoformat()} .. {common_times[-1].isoformat()})",
        file=sys.stderr,
    )

    _plot(
        common_times,
        diff,
        label_a=args.ssa_file.stem,
        label_b=args.gop_tai_file.stem,
        out_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
