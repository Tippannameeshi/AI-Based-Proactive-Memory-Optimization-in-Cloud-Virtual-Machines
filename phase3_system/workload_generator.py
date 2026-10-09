"""Bounded guest workload profiles launched through QEMU Guest Agent."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .live_telemetry_collector import LiveTelemetryCollector, TelemetryError


MAX_MEMORY_MIB_PER_VM = 512
MAX_TOTAL_MEMORY_MIB = 768
MIN_HOST_AVAILABLE_RESERVE_MIB = 2048
MIN_GUEST_AVAILABLE_RESERVE_MIB = 1024

_WORKER = r'''
import gc
import sys
import time

profile = sys.argv[1]
duration = float(sys.argv[2])
limit_mib = int(sys.argv[3])
chunk_mib = 16
blocks = []
allocated_mib = 0
start = time.monotonic()
counter = 0

def desired_memory(elapsed):
    if profile == "memory_burst":
        return limit_mib
    if profile in ("memory_ramp", "mixed"):
        ramp_seconds = max(1.0, duration * 0.6)
        if elapsed < ramp_seconds:
            return int(limit_mib * elapsed / ramp_seconds)
        if elapsed < duration * 0.85:
            return limit_mib
        return int(limit_mib * max(0.0, (duration - elapsed) / max(1.0, duration * 0.15)))
    return 0

while True:
    elapsed = time.monotonic() - start
    if elapsed >= duration:
        break
    target = desired_memory(elapsed)
    while allocated_mib + chunk_mib <= target:
        block = bytearray(chunk_mib * 1024 * 1024)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        blocks.append(block)
        allocated_mib += chunk_mib
    while blocks and allocated_mib > target + chunk_mib:
        blocks.pop()
        allocated_mib -= chunk_mib
    if profile in ("cpu_burst", "mixed"):
        until = time.monotonic() + 0.15
        while time.monotonic() < until:
            counter = (counter * 1664525 + 1013904223) & 0xffffffff
    else:
        time.sleep(0.15)
blocks.clear()
gc.collect()
'''


@dataclass
class GuestWorkload:
    vm_name: str
    pid: int
    profile: str


def validate_workload_request(vm_names: list[str], profile: str, duration: float,
                              memory_mib: int, host_available_mib: float,
                              guest_available_mib: dict[str, float]) -> None:
    valid_profiles = {"idle", "memory_burst", "memory_ramp", "cpu_burst", "mixed"}
    if profile not in valid_profiles:
        raise ValueError(f"Unsupported workload profile: {profile}")
    if duration <= 0:
        raise ValueError("Workload duration must be greater than zero")
    if memory_mib < 0 or memory_mib > MAX_MEMORY_MIB_PER_VM:
        raise ValueError(
            f"memory-mib must be between 0 and {MAX_MEMORY_MIB_PER_VM}"
        )
    if profile in {"memory_burst", "memory_ramp", "mixed"} and memory_mib < 16:
        raise ValueError("Memory workload requires at least one 16 MiB allocation chunk")
    per_vm_memory = (
        memory_mib
        if profile in {"memory_burst", "memory_ramp", "mixed"}
        else 0
    )
    total_requested = len(vm_names) * per_vm_memory
    if total_requested > MAX_TOTAL_MEMORY_MIB:
        raise ValueError(
            f"Selected VMs would request {total_requested} MiB; "
            f"aggregate cap is {MAX_TOTAL_MEMORY_MIB} MiB"
        )
    if (total_requested
            and host_available_mib - total_requested < MIN_HOST_AVAILABLE_RESERVE_MIB):
        raise ValueError(
            "Host MemAvailable is insufficient to preserve the 2 GiB safety reserve"
        )
    if per_vm_memory:
        for vm_name in vm_names:
            available = guest_available_mib.get(vm_name)
            if (available is None
                    or available - per_vm_memory < MIN_GUEST_AVAILABLE_RESERVE_MIB):
                raise ValueError(
                    f"{vm_name} cannot preserve the 1 GiB guest MemAvailable reserve"
                )


def launch_workload(collector: LiveTelemetryCollector, vm_name: str, profile: str,
                    duration: float, memory_mib: int) -> GuestWorkload:
    if profile not in {"memory_burst", "memory_ramp", "cpu_burst", "mixed"}:
        raise ValueError(f"Unsupported guest workload profile: {profile}")
    if duration <= 0:
        raise ValueError("Workload duration must be greater than zero")
    if memory_mib < 0 or memory_mib > MAX_MEMORY_MIB_PER_VM:
        raise ValueError(
            f"memory-mib must be between 0 and {MAX_MEMORY_MIB_PER_VM}"
        )
    if profile in {"memory_burst", "memory_ramp", "mixed"} and memory_mib < 16:
        raise ValueError("Memory workload requires at least one 16 MiB allocation chunk")
    meminfo = collector.guest_meminfo(vm_name)
    validate_workload_request(
        [vm_name],
        profile,
        duration,
        memory_mib,
        host_available_memory_mib(),
        {vm_name: meminfo["MemAvailable"] / 1024.0},
    )
    pid = collector.start_guest_process(
        vm_name,
        "/usr/bin/python3",
        ["-c", _WORKER, profile, str(duration), str(memory_mib)],
        capture_output=False,
        timeout_seconds=5.0,
    )
    if pid <= 0:
        raise TelemetryError(f"{vm_name}: guest Python returned invalid PID {pid}")
    return GuestWorkload(vm_name, pid, profile)


def stop_workload(collector: LiveTelemetryCollector, workload: GuestWorkload,
                  timeout_seconds: float = 3.0) -> None:
    status = collector.guest_process_status(workload.vm_name, workload.pid)
    if status.get("exited"):
        if status.get("exitcode") != 0:
            raise TelemetryError(
                f"{workload.vm_name}: workload PID {workload.pid} exited "
                f"unexpectedly with code {status.get('exitcode')}"
            )
        return
    code, _stdout, stderr = collector.guest_exec(
        workload.vm_name, "/bin/kill", ["-TERM", str(workload.pid)],
        timeout_seconds=timeout_seconds,
    )
    if code not in (0, 1):
        raise TelemetryError(
            f"{workload.vm_name}: could not stop workload PID {workload.pid}: "
            f"{stderr.strip()}"
        )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if collector.guest_process_status(workload.vm_name, workload.pid).get("exited"):
            return
        time.sleep(0.05)
    raise TelemetryError(
        f"{workload.vm_name}: workload PID {workload.pid} did not exit after SIGTERM"
    )


def host_available_memory_mib(path: str = "/proc/meminfo") -> float:
    from pathlib import Path

    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            try:
                return float(line.split()[1]) / 1024.0
            except (IndexError, ValueError) as exc:
                raise TelemetryError("Could not parse host MemAvailable") from exc
    raise TelemetryError("Host /proc/meminfo does not contain MemAvailable")
