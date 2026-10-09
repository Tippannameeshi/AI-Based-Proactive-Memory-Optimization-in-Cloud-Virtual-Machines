"""Resume a balanced live-telemetry campaign across three VMs and five workloads."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .live_telemetry_collector import LIBVIRT_URI, TELEMETRY_COLUMNS, TelemetryError
from .run_live_experiment import DEFAULT_OUTPUT, DEFAULT_VM_NAMES, run_experiment
from .workload_generator import MAX_MEMORY_MIB_PER_VM


WORKLOAD_PROFILES = ("idle", "memory_burst", "memory_ramp", "cpu_burst", "mixed")
DEFAULT_TARGET_ROWS = 20_000


def read_progress(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "rows": 0,
            "vm_counts": Counter(),
            "workload_counts": Counter(),
            "phase_counts": Counter(),
            "pair_counts": Counter(),
            "experiment_ids": set(),
            "pair_experiments": Counter(),
        }
    if path.stat().st_size == 0:
        return {
            "rows": 0,
            "vm_counts": Counter(),
            "workload_counts": Counter(),
            "phase_counts": Counter(),
            "pair_counts": Counter(),
            "experiment_ids": set(),
            "pair_experiments": Counter(),
        }

    vm_counts: Counter[str] = Counter()
    workload_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    experiment_ids: set[str] = set()
    experiment_pairs: dict[str, set[tuple[str, str]]] = {}
    row_count = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TELEMETRY_COLUMNS:
            raise TelemetryError(
                f"Existing CSV has incompatible schema: {path}; refusing to collect"
            )
        for line_number, row in enumerate(reader, start=2):
            if not row.get("experiment_id") or not row.get("vm_id") or not row.get(
                "workload_profile"
            ):
                raise TelemetryError(f"{path}:{line_number}: missing experiment/VM/workload ID")
            row_count += 1
            vm_counts[row["vm_id"]] += 1
            workload_counts[row["workload_profile"]] += 1
            phase_counts[row["phase"]] += 1
            pair_counts[(row["vm_id"], row["workload_profile"])] += 1
            experiment_ids.add(row["experiment_id"])
            experiment_pairs.setdefault(row["experiment_id"], set()).add(
                (row["vm_id"], row["workload_profile"])
            )
    return {
        "rows": row_count,
        "vm_counts": vm_counts,
        "workload_counts": workload_counts,
        "phase_counts": phase_counts,
        "pair_counts": pair_counts,
        "experiment_ids": experiment_ids,
        "pair_experiments": Counter(
            pair for pairs in experiment_pairs.values() for pair in pairs
        ),
    }


def print_progress(path: Path, target_rows: int = DEFAULT_TARGET_ROWS) -> None:
    progress = read_progress(path)
    row_count = int(progress["rows"])
    vm_counts = progress["vm_counts"]
    workload_counts = progress["workload_counts"]
    phase_counts = progress["phase_counts"]
    for label, counts, expected in (
        ("VM", vm_counts, DEFAULT_VM_NAMES),
        ("workload", workload_counts, WORKLOAD_PROFILES),
        ("phase", phase_counts, ("baseline", "workload", "recovery")),
    ):
        print(f"{label} counts:")
        for name in expected:
            print(f"  {name}: {counts.get(name, 0)}")
        for name in sorted(set(counts) - set(expected)):
            print(f"  {name} (unexpected): {counts[name]}")
    print(f"Rows: {row_count:,} / {target_rows:,} ({min(100.0, 100.0 * row_count / target_rows):.1f}%)")
    print(f"Unique experiment IDs: {len(progress['experiment_ids']):,}")
    print(f"Rows remaining: {max(0, target_rows - row_count):,}")


def _next_pair(pair_counts: Counter[tuple[str, str]]) -> tuple[str, str]:
    pairs = [
        (vm_name, profile)
        for vm_name in DEFAULT_VM_NAMES
        for profile in WORKLOAD_PROFILES
    ]
    return min(pairs, key=lambda pair: pair_counts[pair])


def _next_vm_group(
    pair_counts: Counter[tuple[str, str]],
) -> tuple[list[str], str]:
    vm_name, profile = _next_pair(pair_counts)
    least_collected = pair_counts[(vm_name, profile)]
    vm_names = [
        candidate
        for candidate in DEFAULT_VM_NAMES
        if pair_counts[(candidate, profile)] == least_collected
    ]
    return vm_names, profile


def _workload_variant(profile: str, repetition: int, duration: float,
                      memory_cap_mib: int) -> tuple[float, int]:
    duration_options = (
        max(15.0, duration * 0.75),
        duration,
        max(15.0, duration * 1.5),
    )
    memory_options = (
        max(16, int(memory_cap_mib * 0.5 // 16) * 16),
        max(16, int(memory_cap_mib * 0.75 // 16) * 16),
        memory_cap_mib,
    )
    if profile in {"memory_burst", "memory_ramp", "mixed"}:
        return duration_options[repetition % len(duration_options)], memory_options[
            repetition % len(memory_options)
        ]
    return duration_options[repetition % len(duration_options)], 0


def collect_campaign(path: Path, target_rows: int, duration: float,
                     baseline_seconds: float, recovery_seconds: float,
                     memory_mib: int, interval: float,
                     uri: str = LIBVIRT_URI) -> None:
    if target_rows <= 0:
        raise ValueError("Target rows must be greater than zero")
    if duration <= 0:
        raise ValueError("Workload duration must be greater than zero")
    if interval <= 0:
        raise ValueError("Sampling interval must be greater than zero")
    if baseline_seconds < 0 or recovery_seconds < 0:
        raise ValueError("Baseline and recovery durations cannot be negative")
    if memory_mib < 16 or memory_mib > MAX_MEMORY_MIB_PER_VM:
        raise ValueError(
            f"Campaign memory cap must be between 16 and {MAX_MEMORY_MIB_PER_VM} MiB"
        )
    progress = read_progress(path)
    if int(progress["rows"]) >= target_rows:
        print_progress(path, target_rows)
        print("Target already reached; no experiment was started.")
        return

    try:
        while int(progress["rows"]) < target_rows:
            pair_counts = progress["pair_counts"]
            vm_names, profile = _next_vm_group(pair_counts)
            pair_experiments = progress["pair_experiments"]
            repetition = min(
                pair_experiments[(vm_name, profile)] for vm_name in vm_names
            )
            workload_duration, workload_memory = _workload_variant(
                profile, repetition, duration, memory_mib
            )
            before = int(progress["rows"])
            print(
                f"Starting {', '.join(vm_names)} / {profile}; "
                f"duration={workload_duration:g}s, memory={workload_memory} MiB; "
                f"dataset progress {before:,}/{target_rows:,} rows."
            )
            run_experiment(
                vm_names,
                profile,
                workload_duration,
                workload_memory,
                interval,
                baseline_seconds,
                recovery_seconds,
                path,
                uri,
            )
            progress = read_progress(path)
            if int(progress["rows"]) <= before:
                raise TelemetryError(
                    "Experiment appended no rows; stopping rather than looping indefinitely"
                )
            print_progress(path, target_rows)
    except KeyboardInterrupt:
        print("\nCampaign interrupted. Completed rows were preserved; rerun the same "
              "command to resume.")
        print_progress(path, target_rows)
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="seconds per workload phase")
    parser.add_argument("--baseline-seconds", type=float, default=30.0)
    parser.add_argument("--recovery-seconds", type=float, default=30.0)
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--uri", default=LIBVIRT_URI)
    parser.add_argument("--progress", action="store_true",
                        help="report current coverage without collecting")
    args = parser.parse_args(argv)
    if args.progress:
        print_progress(args.output, args.target_rows)
        return 0
    collect_campaign(
        args.output, args.target_rows, args.duration, args.baseline_seconds,
        args.recovery_seconds, args.memory_mib, args.interval, args.uri,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
