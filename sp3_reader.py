#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import re
import sys


# ============================================================
# Constants
# ============================================================

MJD_EPOCH = datetime(1858, 11, 17, 0, 0, 0)

SUPPORTED_TIME_SCALES = {"TAI", "UTC", "GPS"}

_GPS_TAI = 19  # TAI = GPS + 19 s (fixed offset, no leap seconds)

# Leap second table is verified up to this date (IERS Bulletin C, Jan 2026:
# no leap second on 2026-07-01). Data beyond this date may be affected if a
# new leap second is announced.
_LEAPS_KNOWN_UNTIL = datetime(2026, 12, 31)

# TAI − UTC offset in seconds, effective from date (IERS Bulletin C).
# Last leap second: 2017-01-01.  Update when a new one is announced.
_LEAPS: tuple[tuple[datetime, int], ...] = (
    (datetime(1972,  1,  1), 10), (datetime(1972,  7,  1), 11),
    (datetime(1973,  1,  1), 12), (datetime(1974,  1,  1), 13),
    (datetime(1975,  1,  1), 14), (datetime(1976,  1,  1), 15),
    (datetime(1977,  1,  1), 16), (datetime(1978,  1,  1), 17),
    (datetime(1979,  1,  1), 18), (datetime(1980,  1,  1), 19),
    (datetime(1981,  7,  1), 20), (datetime(1982,  7,  1), 21),
    (datetime(1983,  7,  1), 22), (datetime(1985,  7,  1), 23),
    (datetime(1988,  1,  1), 24), (datetime(1990,  1,  1), 25),
    (datetime(1991,  1,  1), 26), (datetime(1992,  7,  1), 27),
    (datetime(1993,  7,  1), 28), (datetime(1994,  7,  1), 29),
    (datetime(1996,  1,  1), 30), (datetime(1997,  7,  1), 31),
    (datetime(1999,  1,  1), 32), (datetime(2006,  1,  1), 33),
    (datetime(2009,  1,  1), 34), (datetime(2012,  7,  1), 35),
    (datetime(2015,  7,  1), 36), (datetime(2017,  1,  1), 37),
)

# SP3 default units:
# P records: km
# V records: dm/s
#
# This script normalizes them immediately to:
# position: m
# velocity: m/s


# ============================================================
# General helpers
# ============================================================

def _safe_float(value: str) -> float | None:
    """Convert SP3 numeric field to float. Returns None for empty or invalid values."""
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_float(value: float | None) -> str:
    """Format a number for text output. Returns empty string if value is missing."""
    return "" if value is None else f"{value:.12g}"


# ============================================================
# Header detection
# ============================================================

def _split_header_and_body(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split SP3 file into header and body. Body starts at the first epoch line ('*')."""
    for i, line in enumerate(lines):
        if line.lstrip().startswith("*"):
            return lines[:i], lines[i:]
    return lines, []


# ============================================================
# Header processing
# ============================================================

_TIME_SCALE_RE = re.compile(r"\b(GPS|UTC|TAI)\b", re.IGNORECASE)


def _extract_coordinate_system_from_header(header_lines: list[str]) -> str:
    """Extract coordinate/reference system from SP3 header. Returns UNKNOWN if not found."""
    candidates = ("ITRF", "IGS", "GCRF", "ICRF", "WGS84", "WGS 84")

    for line in header_lines:
        upper_line = line.upper()
        for candidate in candidates:
            if candidate in upper_line:
                return candidate.replace(" ", "")

    return "UNKNOWN"


def _extract_time_scale_from_header(header_lines: list[str]) -> str:
    """Extract time scale (GPS/UTC/TAI) from SP3 header. Prefers %c lines, falls back to full header."""
    # Prioritise %c lines (official time scale field), then fall back to any match
    priority = [l for l in header_lines if l.lstrip().startswith("%c")]
    for line in priority + header_lines:
        match = _TIME_SCALE_RE.search(line)
        if match:
            return match.group(1).upper()

    return "UNKNOWN"


def _parse_header_line1(header_lines: list[str]) -> dict:
    """
    Parse SP3 first header line (#).

    Example
    -------
    #cP2023  8 20  0  0  0.00000000    2880   d+D   IGS14 FIT  IGS
    """
    for line in header_lines:
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            parts = stripped.split()
            if len(parts) < 11:
                return {}
            prefix = parts[0]
            try:
                year   = int(prefix[3:])
                month  = int(parts[1])
                day    = int(parts[2])
                hour   = int(parts[3])
                minute = int(parts[4])
                second = int(float(parts[5]))
                start_epoch = datetime(year, month, day, hour, minute, second).isoformat()
            except (ValueError, IndexError):
                start_epoch = "UNKNOWN"
            return {
                "version":      prefix[1] if len(prefix) > 1 else "UNKNOWN",
                "pos_vel_flag": prefix[2] if len(prefix) > 2 else "UNKNOWN",
                "start_epoch":  start_epoch,
                "epoch_count":  int(parts[6]) if len(parts) > 6 else None,
                "data_used":    parts[7]  if len(parts) > 7  else "UNKNOWN",
                "orbit_type":   parts[9]  if len(parts) > 9  else "UNKNOWN",
                "agency":       parts[10] if len(parts) > 10 else "UNKNOWN",
            }
    return {}


def _parse_header_line2(header_lines: list[str]) -> dict:
    """
    Parse SP3 second header line (##).

    Example
    -------
    ##  2272      0.000000000    60.00000000   60143  0.0000000000000
    """
    for line in header_lines:
        stripped = line.lstrip()
        if stripped.startswith("##"):
            parts = stripped.split()
            if len(parts) < 6:
                return {}
            try:
                return {
                    "gps_week":               int(parts[1]),
                    "gps_seconds_of_week":    float(parts[2]),
                    "epoch_interval_seconds": float(parts[3]),
                    "start_mjd":              float(parts[4]) + float(parts[5]),
                }
            except (ValueError, IndexError):
                return {}
    return {}


def _parse_header_satellites(header_lines: list[str]) -> list[str]:
    """
    Parse satellite identifiers from SP3 header (+) lines.

    Example
    -------
    +   3   L20  L21  L22  0  0  0  0  0  0  0  0  0  0  0  0  0
    """
    satellites = []
    plus_lines = [l for l in header_lines if l.lstrip().startswith("+") and not l.lstrip().startswith("++")]
    for i, line in enumerate(plus_lines):
        parts = line.lstrip().split()
        satellites += [sat for sat in parts[2 if i == 0 else 1:] if sat not in ("0", "000")]
    return satellites


def _parse_header_comments(header_lines: list[str]) -> list[str]:
    """
    Parse comment lines (/*) from SP3 header.
    """
    return [
        line.lstrip()[2:].strip()
        for line in header_lines
        if line.lstrip().startswith("/*") and line.lstrip()[2:].strip()
    ]


def _parse_sp3_header(header_lines: list[str]) -> dict:
    """
    Parse SP3 header metadata.

    Raw header lines are kept unchanged.
    """
    line1      = _parse_header_line1(header_lines)
    line2      = _parse_header_line2(header_lines)
    satellites = _parse_header_satellites(header_lines)
    comments   = _parse_header_comments(header_lines)

    return {
        "raw_header":             header_lines,
        "version":                line1.get("version",      "UNKNOWN"),
        "pos_vel_flag":           line1.get("pos_vel_flag", "UNKNOWN"),
        "time_scale":             _extract_time_scale_from_header(header_lines),
        "coordinate_system":      _extract_coordinate_system_from_header(header_lines),
        "orbit_type":             line1.get("orbit_type",   "UNKNOWN"),
        "agency":                 line1.get("agency",       "UNKNOWN"),
        "data_used":              line1.get("data_used",    "UNKNOWN"),
        "start_epoch":            line1.get("start_epoch",  "UNKNOWN"),
        "start_mjd":              line2.get("start_mjd"),
        "epoch_count":            line1.get("epoch_count"),
        "epoch_interval_seconds": line2.get("epoch_interval_seconds"),
        "gps_week":               line2.get("gps_week"),
        "satellites":             satellites,
        "comments":               comments,
    }


# ============================================================
# Time handling
# ============================================================

def _datetime_to_mjd(dt: datetime) -> float:
    """
    Convert datetime to Modified Julian Date (MJD).

    Uses integer Julian Day Number formula to avoid
    precision loss from Unix timestamp conversion.
    Precision: ~8.6 nanoseconds (vs ~10 microseconds with timestamp()).
    """
    y = dt.year
    m = dt.month
    d = dt.day

    # Integer Julian Day Number (Meeus formula, exact)
    a  = (14 - m) // 12
    y1 = y + 4800 - a
    m1 = m + 12 * a - 3
    jdn = d + (153 * m1 + 2) // 5 + 365 * y1 + y1 // 4 - y1 // 100 + y1 // 400 - 32045

    # Fractional day from time components
    frac = (dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6) / 86400.0

    # MJD = JD - 2400000.5; JD = JDN - 0.5 + frac
    return jdn - 2400001 + frac


_leaps_warning_issued = False


def _tai_minus_utc_seconds_approx(dt: datetime) -> int:
    """
    Return TAI − UTC offset in seconds for the given datetime.

    Uses module-level _LEAPS table; accurate to the second for dates after 1972.
    Emits a one-time warning if dt is beyond _LEAPS_KNOWN_UNTIL.
    """
    global _leaps_warning_issued
    if not _leaps_warning_issued and dt > _LEAPS_KNOWN_UNTIL:
        print(
            "Warning: data contains epochs beyond 2026-12-31. "
            "Leap seconds after this date are not known and were not applied.",
            file=sys.stderr,
        )
        _leaps_warning_issued = True

    offset = 0
    for start, value in _LEAPS:
        if dt >= start:
            offset = value
        else:
            break
    return offset


def _convert_datetime_between_scales(
    dt: datetime,
    source_scale: str,
    target_scale: str,
) -> datetime:
    """
    Convert datetime between TAI, GPS and UTC.

    Notes
    -----
    GPS = TAI - 19 s
    TAI = UTC + leap seconds
    """
    source = source_scale.upper()
    target = target_scale.upper()

    if source == target:
        return dt

    if source not in SUPPORTED_TIME_SCALES:
        raise ValueError(f"Unsupported source time scale: {source_scale!r}")

    if target not in SUPPORTED_TIME_SCALES:
        raise ValueError(f"Unsupported target time scale: {target_scale!r}")

    # Convert source time to TAI-like time.
    if source == "TAI":
        tai_dt = dt
    elif source == "GPS":
        tai_dt = dt + timedelta(seconds=_GPS_TAI)
    else:  # UTC
        tai_dt = dt + timedelta(seconds=_tai_minus_utc_seconds_approx(dt))

    # Convert TAI-like time to target scale.
    if target == "TAI":
        return tai_dt

    if target == "GPS":
        return tai_dt - timedelta(seconds=_GPS_TAI)

    # UTC
    utc_guess = tai_dt - timedelta(seconds=_tai_minus_utc_seconds_approx(tai_dt))
    utc_offset = _tai_minus_utc_seconds_approx(utc_guess)
    return tai_dt - timedelta(seconds=utc_offset)


def _format_time(
    epoch: datetime,
    *,
    source_time_scale: str,
    output_time_scale: str,
    time_format: str,
) -> str:
    """
    Format epoch as ISO or MJD.

    If the source time scale is unknown, no time-scale conversion is applied.
    """
    if source_time_scale == "UNKNOWN":
        converted_epoch = epoch
    else:
        converted_epoch = _convert_datetime_between_scales(
            epoch,
            source_scale=source_time_scale,
            target_scale=output_time_scale,
        )

    if time_format == "iso":
        return converted_epoch.isoformat()

    if time_format == "mjd":
        return f"{_datetime_to_mjd(converted_epoch):.12f}"

    raise ValueError(f"Unsupported time format: {time_format!r}")


# ============================================================
# Body record processing
# ============================================================

def _parse_epoch_line(line: str) -> datetime:
    """
    Parse SP3 epoch line.

    Example
    -------
    *  2025  1  1  0  0  0.00000000
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

    if microsecond == 1_000_000:
        second += 1
        microsecond = 0

    return datetime(year, month, day, hour, minute, second, microsecond)


def _parse_sp3_record_line(line: str) -> tuple[str, float | None, float | None, float | None]:
    """
    Parse fixed-width SP3 position or velocity record.

    Position input unit is km, velocity input unit is dm/s.
    """
    satellite = line[1:4].strip()
    a = _safe_float(line[4:18])
    b = _safe_float(line[18:32])
    c = _safe_float(line[32:46])

    return satellite, a, b, c


@dataclass
class OrbitRecord:
    epoch:        datetime
    satellite:    str
    x:            float | None = None
    y:            float | None = None
    z:            float | None = None
    vx:           float | None = None
    vy:           float | None = None
    vz:           float | None = None
    has_position: bool = False
    has_velocity: bool = False


# ============================================================
# Unit normalization
# ============================================================

def _scale_value(value: float | None, factor: float) -> float | None:
    """Multiply value by factor, or return None if value is None."""
    return None if value is None else value * factor


def _get_or_create_record(records: dict, epoch: datetime, satellite: str) -> OrbitRecord:
    """Return existing orbit record for (epoch, satellite), or create a new one."""
    key = (epoch, satellite)
    if key not in records:
        records[key] = OrbitRecord(epoch=epoch, satellite=satellite)
    return records[key]


# ============================================================
# SP3 file reading
# ============================================================

def _detect_expected_step(steps: list[float]) -> float | None:
    """
    Detect the expected epoch interval from the data.

    Uses the most common positive step as the reference interval.
    """
    if not steps:
        return None
    counts: dict[float, int] = {}
    for s in steps:
        if s > 0:
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=lambda s: counts[s]) if counts else None


def _classify_step(step_seconds: float, expected: float) -> str | None:
    """
    Classify a single epoch step relative to the expected interval.

    Returns None for normal steps; otherwise returns an issue type string.
    Mirrors the classify logic from inspect_orbit_time_series in the library.
    """
    if step_seconds == expected:
        return None
    if step_seconds == 0:
        return "duplicate"
    if step_seconds < 0:
        return "non_monotonic"
    if step_seconds > expected:
        return "gap"
    return "irregular"


def _read_sp3_file(path: Path) -> tuple[dict, dict[tuple[datetime, str], OrbitRecord], dict]:
    """
    Read SP3 file and return header, records and validation statistics.

    Positions are normalized from km to m.
    Velocities are normalized from dm/s to m/s.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    header_lines, body_lines = _split_header_and_body(lines)
    header = _parse_sp3_header(header_lines)

    current_epoch: datetime | None = None
    records: dict[tuple[datetime, str], OrbitRecord] = {}

    epoch_counts: dict[datetime, int] = {}
    epoch_sequence: list[datetime] = []

    duplicate_position_count = 0
    duplicate_velocity_count = 0

    for raw_line in body_lines:
        line = raw_line.rstrip("\n")
        stripped = line.lstrip()

        if not stripped:
            continue

        record_type = stripped[0].upper()

        if record_type == "*":
            current_epoch = _parse_epoch_line(stripped)
            epoch_sequence.append(current_epoch)
            epoch_counts[current_epoch] = epoch_counts.get(current_epoch, 0) + 1
            continue

        if current_epoch is None:
            continue

        if record_type == "P":
            satellite, a, b, c = _parse_sp3_record_line(stripped)
            rec = _get_or_create_record(records, current_epoch, satellite)
            if rec.has_position:
                duplicate_position_count += 1
            rec.x = _scale_value(a, 1000.0)
            rec.y = _scale_value(b, 1000.0)
            rec.z = _scale_value(c, 1000.0)
            rec.has_position = True
            continue

        if record_type == "V":
            satellite, a, b, c = _parse_sp3_record_line(stripped)
            rec = _get_or_create_record(records, current_epoch, satellite)
            if rec.has_velocity:
                duplicate_velocity_count += 1
            rec.vx = _scale_value(a, 0.1)
            rec.vy = _scale_value(b, 0.1)
            rec.vz = _scale_value(c, 0.1)
            rec.has_velocity = True
            continue

    all_steps = [
        (b - a).total_seconds()
        for a, b in zip(epoch_sequence, epoch_sequence[1:])
    ]
    expected_step = _detect_expected_step(all_steps)

    time_gap_count        = 0
    irregular_step_count  = 0
    non_monotonic_epoch_count = 0

    for step_seconds in all_steps:
        kind = _classify_step(step_seconds, expected_step or 0)
        if kind == "non_monotonic":
            non_monotonic_epoch_count += 1
        elif kind == "gap":
            time_gap_count += 1
        elif kind == "irregular":
            irregular_step_count += 1

    stats = {
        "epoch_count": len(epoch_counts),
        "duplicate_epoch_count": sum(1 for count in epoch_counts.values() if count > 1),
        "duplicate_position_count": duplicate_position_count,
        "duplicate_velocity_count": duplicate_velocity_count,
        "record_count": len(records),
        "missing_position_count": sum(1 for record in records.values() if not record.has_position),
        "missing_velocity_count": sum(1 for record in records.values() if not record.has_velocity),
        "non_monotonic_epoch_count": non_monotonic_epoch_count,
        "time_gap_count":            time_gap_count,
        "irregular_step_count":      irregular_step_count,
        "detected_step_seconds":     expected_step,
    }

    return header, records, stats


# ============================================================
# Validation
# ============================================================

def _print_validation_warnings(stats: dict) -> None:
    """
    Print validation warnings to stderr.

    stdout stays clean for data output and command-line pipelines.
    """
    warnings = [
        ("duplicate_epoch_count",     "duplicate epoch(s) detected"),
        ("duplicate_position_count",  "duplicate position record(s) detected"),
        ("duplicate_velocity_count",  "duplicate velocity record(s) detected"),
        ("missing_position_count",    "record(s) missing position data"),
        ("missing_velocity_count",    "record(s) missing velocity data"),
        ("non_monotonic_epoch_count", "non-monotonic epoch step(s) detected"),
        ("time_gap_count",            "time gap(s) detected in epoch sequence"),
        ("irregular_step_count",      "irregular epoch step(s) detected"),
    ]

    for key, message in warnings:
        if stats[key] > 0:
            print(f"Warning: {stats[key]} {message}.", file=sys.stderr)


def _check_single_satellite(
    records: dict[tuple[datetime, str], OrbitRecord],
) -> bool:
    """
    Error if more than one satellite is present in records.
    Returns True if check passes, False if it fails.
    """
    satellites = sorted({record.satellite for record in records.values()})

    if len(satellites) > 1:
        print(
            f"Error: output contains multiple satellites ({', '.join(satellites)}). "
            "Select exactly one with -s.",
            file=sys.stderr,
        )
        return False
    return True


# ============================================================
# Filtering
# ============================================================

def _parse_satellite_filter(value: str | None) -> set[str] | None:
    """
    Parse comma-separated satellite filter.

    Example
    -------
    G01,G02
    """
    if value is None:
        return None

    satellites = {item.strip() for item in value.split(",") if item.strip()}
    return satellites or None


def _filter_records(
    records: dict[tuple[datetime, str], OrbitRecord],
    satellites: set[str] | None,
) -> dict[tuple[datetime, str], OrbitRecord]:
    """
    Filter records by satellite identifiers.
    """
    if satellites is None:
        return records

    return {
        key: record
        for key, record in records.items()
        if record.satellite in satellites
    }


# ============================================================
# Output helpers
# ============================================================


def _print_header_metadata(header: dict, *, output_file) -> None:
    """
    Print metadata in simple key-value format.
    """
    pos_vel = header.get("pos_vel_flag", "UNKNOWN")
    pos_vel_label = {
        "P": "P  (positions only)",
        "V": "V  (positions + velocities)",
    }.get(pos_vel, pos_vel)

    satellites = header.get("satellites", [])
    sat_key    = f"satellites ({len(satellites)})"
    comments   = header.get("comments", [])

    # Width driven by the longest key — "coordinate system" (17 chars).
    # Format: "  {key}: " left-justified to W chars total before the value.
    W = 20

    def row(key: str, value) -> str:
        return f"  {(key + ':'):<{W}} {value}"

    meta_rows = [
        ("version",           header["version"]),
        ("file type",         pos_vel_label),
        ("agency",            header["agency"]),
        ("orbit type",        header["orbit_type"]),
        ("time scale",        header["time_scale"]),
        ("coordinate system", header["coordinate_system"]),
        ("start epoch",       header["start_epoch"]),
        ("start MJD",         header["start_mjd"]),
        ("epoch count",       header["epoch_count"]),
        ("epoch interval",    f"{header['epoch_interval_seconds']} s"),
        (sat_key,             " ".join(satellites)),
    ]

    for label, value in meta_rows:
        print(row(label, value), file=output_file)

    if comments:
        indent = " " * (W + 3)
        print(row("comments", comments[0]), file=output_file)
        for comment in comments[1:]:
            print(f"{indent}{comment}", file=output_file)


def _print_records(
    records: dict[tuple[datetime, str], OrbitRecord],
    *,
    source_time_scale: str,
    output_time_scale: str,
    time_format: str,
    delimiter: str | None,
    output_file,
    column_headers: bool = False,
) -> None:
    """
    Print normalized orbit records.

    Output columns:
    time x_m y_m z_m vx_mps vy_mps vz_mps
    """
    columns = ["time", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"]
    aligned = delimiter is None

    # Průchod 1: naformátovat všechny řádky
    rows = []
    for key in sorted(records):
        record = records[key]
        row = [
            _format_time(
                record.epoch,
                source_time_scale=source_time_scale,
                output_time_scale=output_time_scale,
                time_format=time_format,
            ),
            _format_float(record.x),
            _format_float(record.y),
            _format_float(record.z),
            _format_float(record.vx),
            _format_float(record.vy),
            _format_float(record.vz),
        ]
        rows.append(row)

    # Průchod 2: tisk
    if aligned:
        col_widths = [len(col) for col in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        if column_headers:
            print(" ".join(col.rjust(col_widths[i]) for i, col in enumerate(columns)), file=output_file)
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

    Without this class:  -f iso|mjd, --time-format iso|mjd
    With this class:     -f, --time-format iso|mjd   Output time format.
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

        # Show metavar only next to the long option: -f, --time-format iso|mjd
        if metavar:
            return ", ".join(shorts + [f"{o} {metavar}" for o in longs])
        return ", ".join(action.option_strings)


# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="sp3_reader",
        usage="sp3_reader [options] SP3_FILE",
        description="Read an SP3 orbit file and write a normalised text table.",
        epilog="""\
output columns:
  time  x_m  y_m  z_m  vx_mps  vy_mps  vz_mps

notes:
  positions converted from km to m
  velocities converted from dm/s to m/s
  validation warnings always printed to stderr

examples:
  sp3_reader file.sp3
  sp3_reader file.sp3 -m              print metadata and exit
  sp3_reader file.sp3 -s L24          select satellite
  sp3_reader file.sp3 -f mjd          MJD time format
  sp3_reader file.sp3 -t TAI          convert time scale to TAI
  sp3_reader file.sp3 -d ,            comma-separated output
  sp3_reader file.sp3 -o out.txt      write to file
  sp3_reader file.sp3 -c              print column headers
  sp3_reader file.sp3 -d ";"          semicolon-separated output""",
        formatter_class=_HelpFormatter,
    )

    parser.add_argument(
        "sp3_file",
        type=Path,
        metavar="SP3_FILE",
        help="Input SP3 file.",
    )

    parser.add_argument(
        "-m",
        "--metadata",
        action="store_true",
        help="Print file metadata and exit.",
    )

    parser.add_argument(
        "-s",
        "--sat",
        metavar="SAT",
        help="Satellite ID to extract (required if file has multiple).",
    )

    parser.add_argument(
        "-f",
        "--time-format",
        choices=["iso", "mjd"],
        default="iso",
        metavar="iso|mjd",
        help="Output time format (default: iso).",
    )

    parser.add_argument(
        "-t",
        "--time-scale",
        choices=["TAI", "UTC", "GPS"],
        metavar="TAI|UTC|GPS",
        help="Output time scale (default: from file header).",
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
        help="Column separator (default: aligned columns).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Output file (default: stdout).",
    )

    return parser


# ============================================================
# Main program
# ============================================================

def main() -> int:
    """
    Program entry point.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    path: Path = args.sp3_file

    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        return 1

    try:
        header, records, stats = _read_sp3_file(path)
    except Exception as exc:
        print(f"Error while reading SP3 file: {exc}", file=sys.stderr)
        return 1

    _print_validation_warnings(stats)

    source_time_scale = header["time_scale"]

    if args.time_scale is None:
        output_time_scale = source_time_scale if source_time_scale != "UNKNOWN" else "UTC"
    else:
        output_time_scale = args.time_scale

    if source_time_scale == "UNKNOWN" and args.time_scale is not None:
        print(
            "Warning: source time scale is UNKNOWN; time-scale conversion cannot be performed.",
            file=sys.stderr,
        )

    records = _filter_records(
        records,
        _parse_satellite_filter(args.sat),
    )

    if not records and not args.metadata:
        print("Warning: no orbit records selected.", file=sys.stderr)

    output_file = sys.stdout
    close_output = False

    if args.output is not None:
        output_file = args.output.open("w", encoding="utf-8", newline="")
        close_output = True

    try:
        if args.metadata:
            _print_header_metadata(
                header,
                output_file=output_file,
            )
            return 0

        if not _check_single_satellite(records):
            return 1

        _print_records(
            records,
            source_time_scale=source_time_scale,
            output_time_scale=output_time_scale,
            time_format=args.time_format,
            delimiter=args.delimiter,
            output_file=output_file,
            column_headers=args.column_headers,
        )

    finally:
        if close_output:
            output_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())