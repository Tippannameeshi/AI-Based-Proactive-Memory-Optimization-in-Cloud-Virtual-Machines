"""Validate collected live telemetry CSV files without fitting or transforming data."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

from .live_telemetry_collector import TELEMETRY_COLUMNS
from .collect_live_dataset import DEFAULT_TARGET_ROWS, WORKLOAD_PROFILES


EXPECTED_VMS = ("testvm1", "testvm2", "testvm3")
EXPECTED_PHASES = ("baseline", "workload", "recovery")
FORBIDDEN_CROSS_VM_FEATURE_TOKENS = (
    "rolling", "roll_mean", "roll_std", "lag_", "donor", "receiver",
)
NUMERIC_COLUMNS = tuple(
    column for column in TELEMETRY_COLUMNS
    if column not in {
        "timestamp", "experiment_id", "vm_id", "workload_profile", "phase",
        "workload_active", "collection_errors",
    }
)


def validate_csv(path: Path, expected_vms: tuple[str, ...] = EXPECTED_VMS
                 ) -> tuple[list[str], dict[str, int], list[float]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    intervals: dict[str, list[float]] = defaultdict(list)
    previous_timestamp: dict[str, datetime] = {}
    previous_experiment: dict[str, str] = {}
    samples_seen: set[tuple[str, str]] = set()
    experiment_profiles: dict[str, str] = {}
    experiment_phases: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        if columns != TELEMETRY_COLUMNS:
            return [f"CSV schema mismatch; expected {TELEMETRY_COLUMNS}, found {columns}"], {}, []
        forbidden = [c for c in columns if any(token in c for token in FORBIDDEN_CROSS_VM_FEATURE_TOKENS)]
        if forbidden:
            errors.append(f"Unexpected derived/cross-VM feature columns: {forbidden}")
        for line_number, row in enumerate(reader, start=2):
            vm_name = row["vm_id"]
            experiment_id = row["experiment_id"]
            sequence_key = (experiment_id, vm_name)
            counts[vm_name] += 1
            if vm_name not in EXPECTED_VMS:
                errors.append(f"line {line_number}: unknown VM ID {vm_name!r}")
            if row["workload_profile"] not in WORKLOAD_PROFILES:
                errors.append(
                    f"line {line_number}: unknown workload {row['workload_profile']!r}"
                )
            if row["phase"] not in EXPECTED_PHASES:
                errors.append(f"line {line_number}: unknown phase {row['phase']!r}")
            if not experiment_id:
                errors.append(f"line {line_number}: experiment_id is empty")
            elif experiment_id in experiment_profiles:
                if experiment_profiles[experiment_id] != row["workload_profile"]:
                    errors.append(
                        f"line {line_number}: experiment ID is reused across workloads"
                    )
            else:
                experiment_profiles[experiment_id] = row["workload_profile"]
            old_phase = experiment_phases.get(sequence_key)
            if old_phase in EXPECTED_PHASES and row["phase"] in EXPECTED_PHASES:
                if EXPECTED_PHASES.index(row["phase"]) < EXPECTED_PHASES.index(old_phase):
                    errors.append(
                        f"line {line_number}: experiment phase order moved backwards; "
                        "possible reused experiment ID"
                    )
            experiment_phases[sequence_key] = row["phase"]
            try:
                timestamp = datetime.fromisoformat(row["timestamp"])
                if timestamp.utcoffset() is None:
                    raise ValueError("timestamp is missing a timezone offset")
            except (TypeError, ValueError):
                errors.append(f"line {line_number}: invalid ISO timestamp")
                timestamp = None
            timestamp_elapsed = None
            same_experiment_as_previous = (
                vm_name in previous_experiment
                and previous_experiment[vm_name] == experiment_id
            )
            if timestamp is not None and vm_name in previous_timestamp:
                timestamp_elapsed = (timestamp - previous_timestamp[vm_name]).total_seconds()
                if timestamp_elapsed <= 0:
                    errors.append(f"line {line_number}: timestamp did not increase for {vm_name}")
                elif same_experiment_as_previous:
                    intervals[vm_name].append(timestamp_elapsed)
            if timestamp is not None:
                previous_timestamp[vm_name] = timestamp
                previous_experiment[vm_name] = experiment_id

            parsed: dict[str, float | None] = {}
            for column in NUMERIC_COLUMNS:
                raw = row[column]
                if raw == "":
                    parsed[column] = None
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    errors.append(f"line {line_number}: non-numeric {column}={raw!r}")
                    parsed[column] = None
                    continue
                if not math.isfinite(value):
                    errors.append(f"line {line_number}: non-finite {column}")
                parsed[column] = value
                if value < 0:
                    errors.append(f"line {line_number}: negative {column}")

            total, available, used = (
                parsed["guest_memory_total_mib"],
                parsed["guest_memory_available_mib"],
                parsed["guest_memory_used_mib"],
            )
            required_metrics = (
                "guest_memory_total_mib", "guest_memory_available_mib",
                "guest_memory_used_mib", "guest_memory_used_pct",
                "guest_swap_total_mib", "guest_swap_free_mib", "guest_swap_used_mib",
                "vcpu_count", "cpu_time_ns", "network_rx_bytes", "network_tx_bytes",
            )
            first_in_experiment = sequence_key not in samples_seen
            samples_seen.add(sequence_key)
            for column in required_metrics:
                if parsed[column] is None:
                    errors.append(f"line {line_number}: required metric {column} is unavailable")
            observed_elapsed = parsed["elapsed_seconds_since_previous_sample"]
            derived_columns = (
                "elapsed_seconds_since_previous_sample", "cpu_utilization_pct",
                "network_rx_bytes_per_sec", "network_tx_bytes_per_sec",
            )
            if first_in_experiment:
                for column in derived_columns:
                    if parsed[column] is not None:
                        errors.append(
                            f"line {line_number}: first sample of an experiment "
                            f"must not invent {column}"
                        )
            else:
                for column in derived_columns:
                    if parsed[column] is None:
                        errors.append(
                            f"line {line_number}: {column} unavailable after first "
                            "sample in experiment"
                        )
            if (observed_elapsed is not None and timestamp_elapsed is not None
                    and same_experiment_as_previous):
                if not math.isclose(
                    observed_elapsed, timestamp_elapsed, rel_tol=0.02, abs_tol=0.25
                ):
                    errors.append(
                        f"line {line_number}: recorded elapsed time does not match timestamps"
                    )
            if total is not None and available is not None and used is not None:
                if available > total:
                    errors.append(f"line {line_number}: MemAvailable exceeds MemTotal")
                if used > total:
                    errors.append(f"line {line_number}: guest memory used exceeds MemTotal")
                if not math.isclose(used, total - available, rel_tol=0.0, abs_tol=0.02):
                    errors.append(f"line {line_number}: guest memory arithmetic mismatch")
                used_pct = parsed["guest_memory_used_pct"]
                if total > 0 and used_pct is not None and not math.isclose(
                    used_pct, 100.0 * used / total, rel_tol=0.0, abs_tol=0.1
                ):
                    errors.append(f"line {line_number}: guest memory percentage mismatch")
            swap_total = parsed["guest_swap_total_mib"]
            swap_free = parsed["guest_swap_free_mib"]
            swap_used = parsed["guest_swap_used_mib"]
            if all(value is not None for value in (swap_total, swap_free, swap_used)):
                if swap_free > swap_total or not math.isclose(
                    swap_used, swap_total - swap_free, rel_tol=0.0, abs_tol=0.02
                ):
                    errors.append(f"line {line_number}: guest swap arithmetic mismatch")
            for percent_column in ("cpu_utilization_pct", "guest_memory_used_pct"):
                value = parsed[percent_column]
                if value is not None and value > 100.0 + 0.5:
                    errors.append(f"line {line_number}: {percent_column} exceeds 100%")
            balloon_current = parsed["balloon_current_mib"]
            balloon_maximum = parsed["balloon_maximum_mib"]
            if (balloon_current is not None and balloon_maximum is not None
                    and balloon_current > balloon_maximum + 0.5):
                errors.append(f"line {line_number}: balloon current exceeds maximum")
            if row["collection_errors"]:
                errors.append(f"line {line_number}: collection error: {row['collection_errors']}")
            for rate_column in (
                "network_rx_bytes_per_sec", "network_tx_bytes_per_sec",
            ):
                value = parsed[rate_column]
                if value is not None and value < 0:
                    errors.append(f"line {line_number}: negative network rate")

    if not counts:
        errors.append("CSV contains no samples")
    for vm_name in expected_vms:
        if counts[vm_name] < 2:
            errors.append(
                f"{vm_name} has {counts[vm_name]} samples; at least 2 are required"
            )
    flattened_intervals = [
        interval for vm_intervals in intervals.values() for interval in vm_intervals
    ]
    return errors, dict(counts), flattened_intervals


def print_dataset_report(path: Path, target_rows: int = DEFAULT_TARGET_ROWS) -> None:
    if target_rows <= 0:
        raise ValueError("Target rows must be greater than zero")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    print(f"Data rows: {len(rows):,}")
    print(f"Target: {target_rows:,} ({min(100.0, len(rows) * 100.0 / target_rows):.1f}%)")
    for field, title, expected in (
        ("vm_id", "VM", EXPECTED_VMS),
        ("workload_profile", "Workload", WORKLOAD_PROFILES),
        ("phase", "Phase", ("baseline", "workload", "recovery")),
    ):
        counts = Counter(row[field] for row in rows)
        print(f"{title} counts: " + ", ".join(
            f"{name}={counts.get(name, 0)}" for name in expected
        ))
    print(
        "Unique experiment IDs: "
        f"{len({row['experiment_id'] for row in rows if row['experiment_id']}):,}"
    )
    errors = sum(bool(row["collection_errors"]) for row in rows)
    print(f"Rows with collection errors: {errors:,}")
    missing = Counter(
        column for row in rows for column in TELEMETRY_COLUMNS
        if column != "collection_errors" and row[column] == ""
    )
    missing_columns = [
        f"{column}={count}" for column, count in missing.items()
    ]
    print("Missing values: " + (", ".join(missing_columns) if missing_columns else "none"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--vm", action="append",
        help="VMs expected in this file; defaults to all three configured guests",
    )
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    args = parser.parse_args(argv)
    try:
        expected = tuple(args.vm) if args.vm else EXPECTED_VMS
        errors, counts, intervals = validate_csv(args.csv_path, expected)
        print_dataset_report(args.csv_path, args.target_rows)
    except OSError as exc:
        parser.error(str(exc))
    for vm_name in expected:
        print(f"{vm_name}: {counts.get(vm_name, 0)} samples")
    if intervals:
        print(
            "Observed within-experiment sampling interval seconds: "
            f"min={min(intervals):.3f}, median={median(intervals):.3f}, "
            f"max={max(intervals):.3f}"
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Telemetry CSV validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
