#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
import re
import sys


# ============================================================
# General helpers
# ============================================================

def _safe_float(value: str) -> float | None:
    """
    Convert SP3 numeric field to float, handling blank or invalid values.
    """
    value = value.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


# ============================================================
# Header detection
# ============================================================

def _split_header_and_body(lines: list[str]) -> tuple[list[str], list[str]]:
    """
    Split SP3 file into header lines and body lines.

    Header is assumed to end before the first epoch line starting with '*'.
    """
    for i, line in enumerate(lines):
        if line.lstrip().startswith("*"):
            return lines[:i], lines[i:]

    return lines, []


# ============================================================
# Header processing
# ============================================================

_TIME_SCALE_RE = re.compile(r"\b(GPS|UTC|TAI)\b", re.IGNORECASE)


def _extract_time_scale_from_header(header_lines: list[str]) -> str:
    """
    Extract time scale from SP3 header.

    Strategy
    --------
    1. Look first at lines starting with '%c'
    2. Fallback to the rest of the header
    3. Return UNKNOWN if not found
    """
    for line in header_lines:
        if line.lstrip().startswith("%c"):
            match = _TIME_SCALE_RE.search(line)
            if match:
                return match.group(1).upper()

    for line in header_lines:
        match = _TIME_SCALE_RE.search(line)
        if match:
            return match.group(1).upper()

    return "UNKNOWN"


def _parse_sp3_header(header_lines: list[str]) -> dict:
    """
    Parse basic SP3 header metadata.

    Raw header lines are kept unchanged.
    """
    return {
        "raw_header": header_lines,
        "time_scale": _extract_time_scale_from_header(header_lines),
    }


# ============================================================
# Body record detection
# ============================================================

# Reserved for later body record inspection helpers.


# ============================================================
# Body record processing
# ============================================================

def _parse_epoch_line(line: str) -> datetime:
    """
    Parse SP3 epoch line.

    Example
    -------
    *  2023  8 20  0  0  0.00000000
    """
    parts = line.split()

    if len(parts) < 7:
        raise ValueError(f"Invalid epoch line: {line!r}")

    year = int(parts[1])
    month = int(parts[2])
    day = int(parts[3])
    hour = int(parts[4])
    minute = int(parts[5])
    second_float = float(parts[6])

    second = int(second_float)
    microsecond = int(round((second_float - second) * 1_000_000))

    return datetime(year, month, day, hour, minute, second, microsecond)


def _parse_sp3_position_line(line: str) -> tuple[str, float | None, float | None, float | None]:
    """
    Parse fixed-width SP3 position record.
    """
    satellite = line[1:4].strip()
    x = _safe_float(line[4:18])
    y = _safe_float(line[18:32])
    z = _safe_float(line[32:46])
    return satellite, x, y, z


def _parse_sp3_velocity_line(line: str) -> tuple[str, float | None, float | None, float | None]:
    """
    Parse fixed-width SP3 velocity record.
    """
    satellite = line[1:4].strip()
    vx = _safe_float(line[4:18])
    vy = _safe_float(line[18:32])
    vz = _safe_float(line[32:46])
    return satellite, vx, vy, vz


# ============================================================
# SP3 file reading
# ============================================================

def _read_sp3_file(path: Path) -> tuple[dict, dict[tuple[datetime, str], dict]]:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    header_lines, body_lines = _split_header_and_body(lines)
    header = _parse_sp3_header(header_lines)

    current_epoch: datetime | None = None
    records: dict[tuple[datetime, str], dict] = {}

    for raw_line in body_lines:
        line = raw_line.rstrip("\n")
        stripped = line.lstrip()

        if not stripped:
            continue

        record_type = stripped[0].upper()

        if record_type == "*":
            current_epoch = _parse_epoch_line(stripped)
            continue

        if current_epoch is None:
            continue

        if record_type == "P":
            satellite, x, y, z = _parse_sp3_position_line(stripped)
            key = (current_epoch, satellite)

            if key not in records:
                records[key] = {
                    "epoch": current_epoch,
                    "satellite": satellite,
                    "x": None,
                    "y": None,
                    "z": None,
                    "vx": None,
                    "vy": None,
                    "vz": None,
                }

            records[key]["x"] = x
            records[key]["y"] = y
            records[key]["z"] = z
            continue

        if record_type == "V":
            satellite, vx, vy, vz = _parse_sp3_velocity_line(stripped)
            key = (current_epoch, satellite)

            if key not in records:
                records[key] = {
                    "epoch": current_epoch,
                    "satellite": satellite,
                    "x": None,
                    "y": None,
                    "z": None,
                    "vx": None,
                    "vy": None,
                    "vz": None,
                }

            records[key]["vx"] = vx
            records[key]["vy"] = vy
            records[key]["vz"] = vz
            continue

    return header, records


# ============================================================
# Output
# ============================================================

def _print_raw_header(header: dict) -> None:
    print("# Raw SP3 header")
    for line in header["raw_header"]:
        print(line.rstrip("\n"))


def _print_header_metadata(header: dict) -> None:
    print("# SP3 header metadata")
    print(f"# time_scale: {header['time_scale']}")


def _print_records(records: dict[tuple[datetime, str], dict]) -> None:
    print("time,sat,x,y,z,vx,vy,vz")

    for key in sorted(records):
        record = records[key]

        print(
            f"{record['epoch'].isoformat()},"
            f"{record['satellite']},"
            f"{record['x']},"
            f"{record['y']},"
            f"{record['z']},"
            f"{record['vx']},"
            f"{record['vy']},"
            f"{record['vz']}"
        )


# ============================================================
# CLI
# ============================================================

def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: sp3_reader.py <sp3_file>")
        return 1

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"Error: file not found: {path}")
        return 1

    header, records = _read_sp3_file(path)

    _print_raw_header(header)
    print()
    _print_header_metadata(header)
    print()
    _print_records(records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())