"""Collect baseline, controlled workload, and recovery telemetry."""

from __future__ import annotations

import argparse
import csv
import fcntl
import sys
import time
from pathlib import Path
from typing import Any
import uuid

from .live_telemetry_collector import (
    LIBVIRT_URI,
    TELEMETRY_COLUMNS,
    LiveTelemetryCollector,
    TelemetryError,
)
from .workload_generator import (
    MAX_MEMORY_MIB_PER_VM,
    GuestWorkload,
    host_available_memory_mib,
    launch_workload,
    stop_workload,
    validate_workload_request,
)


DEFAULT_VM_NAMES = ("testvm1", "testvm2", "testvm3")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "live_telemetry.csv"


class TelemetryCsvWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_stream = path.with_suffix(path.suffix + ".lock").open("a")
        fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX)
        exists_and_nonempty = path.exists() and path.stat().st_size > 0
        self.stream = path.open("a+", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=TELEMETRY_COLUMNS)
        if exists_and_nonempty:
            self.stream.seek(0)
            reader = csv.reader(self.stream)
            header = next(reader, [])
            if tuple(header) != TELEMETRY_COLUMNS:
                self.stream.close()
                self.lock_stream.close()
                raise TelemetryError(
                    f"Existing CSV has incompatible schema: {path}; refusing to append"
                )
            self.stream.seek(0, 2)
        else:
            self.writer.writeheader()
            self.stream.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()
        self.lock_stream.close()


def _collect_phase(collector: LiveTelemetryCollector, writer: TelemetryCsvWriter,
                   vm_names: list[str], phase: str, profile: str,
                   duration: float, interval: float,
                   experiment_id: str, active: bool = False,
                   delay_first_sample: bool = False) -> None:
    deadline = time.monotonic() + duration
    next_sample = time.monotonic() + (interval if delay_first_sample else 0.0)
    first = True
    while first or time.monotonic() < deadline:
        first = False
        delay = next_sample - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        for vm_name in vm_names:
            row = collector.sample(
                vm_name, workload_profile=profile, phase=phase,
                workload_active=active, experiment_id=experiment_id,
            )
            writer.write(row)
            if row["collection_errors"]:
                print(
                    f"WARNING {vm_name}: {row['collection_errors']}",
                    file=sys.stderr,
                )
        cycle_finished = time.monotonic()
        if cycle_finished >= deadline:
            break
        next_sample = cycle_finished + interval


def _stop_all_workloads(collector: LiveTelemetryCollector,
                        workloads: list[GuestWorkload]) -> None:
    failures = []
    for workload in workloads:
        try:
            stop_workload(collector, workload)
        except Exception as exc:
            failures.append(f"{workload.vm_name} PID {workload.pid}: {exc}")
    if failures:
        raise TelemetryError("Workload cleanup failed: " + "; ".join(failures))


def _new_experiment_id(path: Path) -> str:
    existing: set[str] = set()
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != TELEMETRY_COLUMNS:
                raise TelemetryError(
                    f"Existing CSV has incompatible schema: {path}; refusing to append"
                )
            existing = {
                row["experiment_id"] for row in reader if row.get("experiment_id")
            }
    experiment_id = uuid.uuid4().hex
    while experiment_id in existing:
        experiment_id = uuid.uuid4().hex
    return experiment_id


def run_experiment(vm_names: list[str], profile: str, duration: float,
                   memory_mib: int, interval: float, baseline_seconds: float,
                   recovery_seconds: float, output: Path,
                   uri: str = LIBVIRT_URI) -> None:
    if interval <= 0:
        raise ValueError("Sampling interval must be greater than zero")
    if baseline_seconds < 0 or recovery_seconds < 0:
        raise ValueError("Baseline and recovery durations cannot be negative")
    if len(set(vm_names)) != len(vm_names):
        raise ValueError("VM names must not contain duplicates")
    workloads: list[GuestWorkload] = []
    writer: TelemetryCsvWriter | None = None
    try:
        with LiveTelemetryCollector(vm_names, uri=uri) as collector:
            guest_available = {}
            for vm_name in vm_names:
                meminfo = collector.guest_meminfo(vm_name)
                guest_available[vm_name] = meminfo["MemAvailable"] / 1024.0
            validate_workload_request(
                vm_names, profile, duration, memory_mib,
                host_available_memory_mib(), guest_available,
            )
            writer = TelemetryCsvWriter(output)
            experiment_id = _new_experiment_id(output)
            print(f"Experiment {experiment_id}: {', '.join(vm_names)}")
            print("Collecting baseline; no VM memory settings are changed.")
            _collect_phase(
                collector, writer, vm_names, "baseline", profile,
                baseline_seconds, interval, experiment_id,
            )
            if profile != "idle":
                try:
                    refreshed_guest_available = {
                        vm_name: collector.guest_meminfo(vm_name)["MemAvailable"] / 1024.0
                        for vm_name in vm_names
                    }
                    validate_workload_request(
                        vm_names, profile, duration, memory_mib,
                        host_available_memory_mib(), refreshed_guest_available,
                    )
                    for vm_name in vm_names:
                        workloads.append(launch_workload(
                            collector, vm_name, profile, duration, memory_mib
                        ))
                    _collect_phase(
                        collector, writer, vm_names, "workload", profile,
                        duration, interval, experiment_id, active=True,
                        delay_first_sample=True,
                    )
                finally:
                    _stop_all_workloads(collector, workloads)
            else:
                _collect_phase(
                    collector, writer, vm_names, "workload", profile,
                    duration, interval, experiment_id, active=False,
                    delay_first_sample=True,
                )
            print("Collecting recovery.")
            _collect_phase(
                collector, writer, vm_names, "recovery", profile,
                recovery_seconds, interval, experiment_id,
                delay_first_sample=True,
            )
    finally:
        if writer is not None:
            writer.close()
    print(f"Appended telemetry to {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", action="append", choices=DEFAULT_VM_NAMES,
                        help="VM to include; repeat to select more than one (default: all)")
    parser.add_argument("--workload", "--profile", dest="profile", choices=(
        "idle", "memory_burst", "memory_ramp", "cpu_burst", "mixed"
    ), default="idle")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="workload phase duration in seconds")
    parser.add_argument("--memory-mib", type=int, default=256,
                        help=f"per-VM cap, maximum {MAX_MEMORY_MIB_PER_VM} MiB")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="requested sampling interval in seconds")
    parser.add_argument("--baseline-seconds", type=float, default=30.0)
    parser.add_argument("--recovery-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--uri", default=LIBVIRT_URI)
    args = parser.parse_args(argv)
    vm_names = args.vm or list(DEFAULT_VM_NAMES)
    run_experiment(
        vm_names, args.profile, args.duration, args.memory_mib, args.interval,
        args.baseline_seconds, args.recovery_seconds, args.output, args.uri,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
