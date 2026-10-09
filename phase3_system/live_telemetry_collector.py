"""Read-only live telemetry collection for running libvirt guests."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Any
import xml.etree.ElementTree as ET


LIBVIRT_URI = "qemu:///system"
MIB = 1024.0
TELEMETRY_COLUMNS = (
    "timestamp",
    "experiment_id",
    "vm_id",
    "workload_profile",
    "phase",
    "workload_active",
    "elapsed_seconds_since_previous_sample",
    "guest_memory_total_mib",
    "guest_memory_available_mib",
    "guest_memory_used_mib",
    "guest_memory_used_pct",
    "guest_swap_total_mib",
    "guest_swap_free_mib",
    "guest_swap_used_mib",
    "balloon_current_mib",
    "balloon_maximum_mib",
    "balloon_unused_mib",
    "balloon_available_mib",
    "balloon_usable_mib",
    "balloon_rss_mib",
    "balloon_swap_in_mib",
    "balloon_swap_out_mib",
    "vcpu_count",
    "cpu_time_ns",
    "cpu_utilization_pct",
    "network_rx_bytes",
    "network_tx_bytes",
    "network_rx_bytes_per_sec",
    "network_tx_bytes_per_sec",
    "collection_errors",
)


class TelemetryError(RuntimeError):
    """A required live telemetry operation failed."""


def parse_meminfo(text: str) -> dict[str, float]:
    """Parse Linux /proc/meminfo values into KiB, rejecting missing key fields."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        values[key] = value

    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    missing = [key for key in required if key not in values]
    if missing:
        raise TelemetryError(f"/proc/meminfo is missing: {', '.join(missing)}")
    if values["MemTotal"] <= 0:
        raise TelemetryError("/proc/meminfo reports a non-positive MemTotal")
    return values


def counter_rate(previous: int | float | None, current: int | float | None,
                 elapsed_seconds: float | None) -> float | None:
    if previous is None or current is None or elapsed_seconds is None:
        return None
    if elapsed_seconds <= 0 or current < previous:
        return None
    return (current - previous) / elapsed_seconds


def normalized_cpu_percent(previous_ns: int | None, current_ns: int,
                           elapsed_seconds: float | None,
                           vcpu_count: int | None) -> float | None:
    if (previous_ns is None or elapsed_seconds is None or elapsed_seconds <= 0
            or vcpu_count is None or vcpu_count <= 0 or current_ns < previous_ns):
        return None
    return 100.0 * ((current_ns - previous_ns) / 1_000_000_000) / (
        elapsed_seconds * vcpu_count
    )


def _decode_guest_output(encoded: str | None) -> str:
    import base64

    if not encoded:
        return ""
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8", errors="replace")
    except (ValueError, UnicodeError) as exc:
        raise TelemetryError(f"QEMU Guest Agent returned invalid base64 output: {exc}") from exc


class LiveTelemetryCollector:
    """Collect guest memory, libvirt CPU/balloon, and interface counters."""

    def __init__(self, vm_names: list[str], uri: str = LIBVIRT_URI,
                 connection: Any | None = None, guest_timeout_seconds: float = 3.0):
        if not vm_names:
            raise ValueError("At least one VM name is required")
        self.vm_names = list(dict.fromkeys(vm_names))
        self.uri = uri
        self.guest_timeout_seconds = guest_timeout_seconds
        self._owns_connection = connection is None
        if connection is None:
            try:
                import libvirt
            except ImportError as exc:
                raise TelemetryError("python-libvirt is required for live collection") from exc
            connection = libvirt.openReadOnly(uri)
            if connection is None:
                raise TelemetryError(f"Could not open read-only libvirt connection: {uri}")
        self.connection = connection
        self.domains = {}
        for name in self.vm_names:
            try:
                domain = self.connection.lookupByName(name)
            except Exception as exc:
                self.close()
                raise TelemetryError(f"Cannot find libvirt domain {name}: {exc}") from exc
            if not domain.isActive():
                self.close()
                raise TelemetryError(f"Domain {name} is not running; refusing to start it")
            self.domains[name] = domain

        self._previous: dict[str, dict[str, Any]] = {}
        self._last_sample_monotonic: dict[str, float] = {}
        self._interfaces: dict[str, list[str]] = {}
        for name, domain in self.domains.items():
            try:
                root = ET.fromstring(domain.XMLDesc(0))
                self._interfaces[name] = [
                    target.get("dev") for target in root.findall("./devices/interface/target")
                    if target.get("dev")
                ]
            except Exception as exc:
                self._interfaces[name] = []
                self._interface_discovery_errors = getattr(
                    self, "_interface_discovery_errors", {}
                )
                self._interface_discovery_errors[name] = str(exc)

    def close(self) -> None:
        if self._owns_connection and getattr(self, "connection", None) is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None

    def __enter__(self) -> "LiveTelemetryCollector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def guest_exec(self, vm_name: str, path: str, args: list[str] | None = None,
                   timeout_seconds: float | None = None) -> tuple[int, str, str]:
        """Execute one fixed-argument program through QGA and return exit/stdout/stderr."""
        timeout = timeout_seconds or self.guest_timeout_seconds
        pid = self.start_guest_process(
            vm_name, path, args or [], capture_output=True,
            timeout_seconds=timeout,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status_response = self._agent_command(
                    vm_name,
                    {
                        "execute": "guest-exec-status",
                        "arguments": {"pid": pid},
                    },
                    max(1, min(2, deadline - time.monotonic())),
                )
                status = status_response["return"]
            except Exception as exc:
                raise TelemetryError(
                    f"{vm_name}: QGA guest-exec-status failed for PID {pid}: {exc}"
                ) from exc
            if status.get("exited"):
                exit_code = status.get("exitcode")
                if exit_code is None:
                    raise TelemetryError(
                        f"{vm_name}: QGA process {pid} exited without an exit code"
                    )
                return (
                    int(exit_code),
                    _decode_guest_output(status.get("out-data")),
                    _decode_guest_output(status.get("err-data")),
                )
            time.sleep(0.05)
        raise TelemetryError(
            f"{vm_name}: guest process {pid} did not finish within {timeout:g} seconds"
        )

    def start_guest_process(self, vm_name: str, path: str, args: list[str],
                            capture_output: bool = False,
                            timeout_seconds: float | None = None) -> int:
        """Start a guest process and return its QGA PID without waiting for exit."""
        timeout = timeout_seconds or self.guest_timeout_seconds
        request = {
            "execute": "guest-exec",
            "arguments": {
                "path": path,
                "arg": args,
                "capture-output": capture_output,
            },
        }
        try:
            response = self._agent_command(vm_name, request, timeout)
            return int(response["return"]["pid"])
        except Exception as exc:
            raise TelemetryError(f"{vm_name}: QGA guest-exec failed for {path}: {exc}") from exc

    def _agent_command(self, vm_name: str, request: dict[str, Any],
                       timeout_seconds: float) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [
                    "virsh", "-c", self.uri, "qemu-agent-command", vm_name,
                    json.dumps(request),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TelemetryError(
                f"{vm_name}: QGA command timed out after {timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise TelemetryError(f"Could not execute virsh QGA command: {exc}") from exc
        if result.returncode != 0:
            raise TelemetryError(
                f"{vm_name}: virsh qemu-agent-command failed: {result.stderr.strip()}"
            )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TelemetryError(
                f"{vm_name}: QGA returned invalid JSON: {result.stdout[:200]!r}"
            ) from exc
        if not isinstance(response, dict) or "return" not in response:
            raise TelemetryError(f"{vm_name}: QGA response lacks a return value")
        return response

    def guest_process_status(self, vm_name: str, pid: int) -> dict[str, Any]:
        try:
            response = self._agent_command(
                vm_name,
                {
                    "execute": "guest-exec-status",
                    "arguments": {"pid": pid},
                },
                2,
            )
            return response["return"]
        except Exception as exc:
            raise TelemetryError(
                f"{vm_name}: could not query QGA process {pid}: {exc}"
            ) from exc

    def guest_meminfo(self, vm_name: str) -> dict[str, float]:
        code, stdout, stderr = self.guest_exec(vm_name, "/bin/cat", ["/proc/meminfo"])
        if code != 0:
            raise TelemetryError(
                f"{vm_name}: /proc/meminfo read exited {code}: {stderr.strip()}"
            )
        return parse_meminfo(stdout)

    def _network_counters(self, vm_name: str) -> tuple[int | None, int | None]:
        errors = []
        devices = self._interfaces.get(vm_name, [])
        rx_total = 0
        tx_total = 0
        for device in devices:
            try:
                stats = self.domains[vm_name].interfaceStats(device)
                if len(stats) < 5:
                    raise TelemetryError(f"short interfaceStats result for {device}")
                rx_total += stats[0]
                tx_total += stats[4]
            except Exception as exc:
                errors.append(f"{device}: {exc}")
        if errors:
            raise TelemetryError(
                f"{vm_name}: interface counter collection failed: {'; '.join(errors)}"
            )
        if not devices:
            if getattr(self, "_interface_discovery_errors", {}).get(vm_name):
                raise TelemetryError(
                    f"{vm_name}: interface discovery failed: "
                    f"{self._interface_discovery_errors[vm_name]}"
                )
            return None, None
        return rx_total, tx_total

    def sample(self, vm_name: str, workload_profile: str = "idle",
               phase: str = "baseline", workload_active: bool = False,
               experiment_id: str = "") -> dict[str, Any]:
        if vm_name not in self.domains:
            raise ValueError(f"VM {vm_name} was not selected")
        domain = self.domains[vm_name]
        errors: list[str] = []
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "vm_id": vm_name,
            "workload_profile": workload_profile,
            "phase": phase,
            "workload_active": bool(workload_active),
            "elapsed_seconds_since_previous_sample": None,
        }

        try:
            meminfo = self.guest_meminfo(vm_name)
            total = meminfo["MemTotal"] / MIB
            available = meminfo["MemAvailable"] / MIB
            used = (meminfo["MemTotal"] - meminfo["MemAvailable"]) / MIB
            swap_total = meminfo["SwapTotal"] / MIB
            swap_free = meminfo["SwapFree"] / MIB
            row.update({
                "guest_memory_total_mib": total,
                "guest_memory_available_mib": available,
                "guest_memory_used_mib": used,
                "guest_memory_used_pct": 100.0 * used / total,
                "guest_swap_total_mib": swap_total,
                "guest_swap_free_mib": swap_free,
                "guest_swap_used_mib": swap_total - swap_free,
            })
        except TelemetryError as exc:
            errors.append(str(exc))
            row.update({key: None for key in (
                "guest_memory_total_mib", "guest_memory_available_mib",
                "guest_memory_used_mib",
                "guest_memory_used_pct", "guest_swap_total_mib", "guest_swap_free_mib",
                "guest_swap_used_mib",
            )})

        try:
            info = domain.info()
            row.update({
                "balloon_maximum_mib": info[1] / MIB,
                "vcpu_count": int(info[3]),
            })
        except Exception as exc:
            errors.append(f"{vm_name}: libvirt memory/vCPU stats failed: {exc}")
            row.update({key: None for key in (
                "balloon_maximum_mib", "vcpu_count",
            )})

        try:
            stats = domain.memoryStats()
            row.update({
                "balloon_current_mib": (
                    stats["actual"] / MIB if "actual" in stats else None
                ),
                "balloon_unused_mib": (
                    stats["unused"] / MIB if "unused" in stats else None
                ),
                "balloon_available_mib": (
                    stats["available"] / MIB if "available" in stats else None
                ),
                "balloon_usable_mib": (
                    stats["usable"] / MIB if "usable" in stats else None
                ),
                "balloon_rss_mib": stats["rss"] / MIB if "rss" in stats else None,
                "balloon_swap_in_mib": (
                    stats["swap_in"] / MIB if "swap_in" in stats else None
                ),
                "balloon_swap_out_mib": (
                    stats["swap_out"] / MIB if "swap_out" in stats else None
                ),
            })
        except Exception as exc:
            errors.append(f"{vm_name}: balloon statistics unavailable: {exc}")
            row.update({key: None for key in (
                "balloon_current_mib", "balloon_unused_mib", "balloon_available_mib",
                "balloon_usable_mib", "balloon_rss_mib", "balloon_swap_in_mib",
                "balloon_swap_out_mib",
            )})

        try:
            cpu_stats = domain.getCPUStats(True)
            cpu_time_ns = int(cpu_stats[0]["cpu_time"])
            row["cpu_time_ns"] = cpu_time_ns
        except Exception as exc:
            cpu_time_ns = None
            row["cpu_time_ns"] = None
            errors.append(f"{vm_name}: CPU counter collection failed: {exc}")

        try:
            rx_bytes, tx_bytes = self._network_counters(vm_name)
            row["network_rx_bytes"] = rx_bytes
            row["network_tx_bytes"] = tx_bytes
        except TelemetryError as exc:
            rx_bytes = tx_bytes = None
            row["network_rx_bytes"] = row["network_tx_bytes"] = None
            errors.append(str(exc))

        sampled_at = time.monotonic()
        previous = self._previous.get(vm_name, {})
        elapsed = (
            sampled_at - self._last_sample_monotonic[vm_name]
            if vm_name in self._last_sample_monotonic else None
        )
        row["elapsed_seconds_since_previous_sample"] = elapsed
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        row["cpu_utilization_pct"] = normalized_cpu_percent(
            previous.get("cpu_time_ns"), cpu_time_ns, elapsed, row.get("vcpu_count")
        )
        row["network_rx_bytes_per_sec"] = counter_rate(
            previous.get("network_rx_bytes"), rx_bytes, elapsed
        )
        row["network_tx_bytes_per_sec"] = counter_rate(
            previous.get("network_tx_bytes"), tx_bytes, elapsed
        )
        self._previous[vm_name] = {
            "cpu_time_ns": cpu_time_ns,
            "network_rx_bytes": rx_bytes,
            "network_tx_bytes": tx_bytes,
        }
        self._last_sample_monotonic[vm_name] = sampled_at
        row["collection_errors"] = " | ".join(errors)
        return row
