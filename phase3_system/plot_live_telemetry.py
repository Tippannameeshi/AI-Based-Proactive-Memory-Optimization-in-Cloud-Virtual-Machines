"""Create plots from collected live telemetry CSV data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .run_live_experiment import DEFAULT_OUTPUT


DEFAULT_PLOT_DIR = Path(__file__).resolve().parent / "data" / "plots"


def load_rows(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("timestamp"):
                row["_time"] = datetime.fromisoformat(row["timestamp"]).isoformat()
                rows[(row["experiment_id"], row["vm_id"])].append(row)
    for vm_rows in rows.values():
        vm_rows.sort(key=lambda row: row["_time"])
    return dict(rows)


def create_plots(csv_path: Path, output_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib; install the declared phase3 requirement"
        ) from exc

    rows_by_vm = load_rows(csv_path)
    if not rows_by_vm:
        raise ValueError(f"No timestamped telemetry rows in {csv_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("guest_memory_used_mib", "Guest memory used (MiB)", "guest_memory_used_mib.png"),
        ("guest_memory_available_mib", "Guest MemAvailable (MiB)",
         "guest_memory_available_mib.png"),
        ("cpu_utilization_pct", "CPU utilization (% of total vCPU capacity)",
         "cpu_utilization.png"),
    ]
    written: list[Path] = []
    for column, ylabel, filename in plots:
        fig, axis = plt.subplots(figsize=(11, 5))
        for (experiment_id, vm_name), rows in sorted(rows_by_vm.items()):
            dates = [datetime.fromisoformat(row["_time"]) for row in rows]
            values = [
                float(row[column]) if row.get(column, "") != "" else float("nan")
                for row in rows
            ]
            axis.plot(dates, values, label=f"{vm_name} ({experiment_id})")
            active_dates = [
                date for date, row in zip(dates, rows)
                if row.get("workload_active", "").lower() == "true"
            ]
            if active_dates:
                axis.axvspan(min(active_dates), max(active_dates), alpha=0.06)
        axis.set_xlabel("Time (UTC)")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        fig.tight_layout()
        destination = output_dir / filename
        fig.savefig(destination, dpi=140)
        plt.close(fig)
        written.append(destination)

    fig, axis = plt.subplots(figsize=(11, 5))
    for (experiment_id, vm_name), rows in sorted(rows_by_vm.items()):
        dates = [datetime.fromisoformat(row["_time"]) for row in rows]
        for column, direction in (
            ("network_rx_bytes_per_sec", "RX"),
            ("network_tx_bytes_per_sec", "TX"),
        ):
            values = [
                float(row[column]) if row.get(column, "") != "" else float("nan")
                for row in rows
            ]
            axis.plot(dates, values, label=f"{vm_name} {direction} ({experiment_id})")
        active_dates = [
            date for date, row in zip(dates, rows)
            if row.get("workload_active", "").lower() == "true"
        ]
        if active_dates:
            axis.axvspan(min(active_dates), max(active_dates), alpha=0.06)
    axis.set_xlabel("Time (UTC)")
    axis.set_ylabel("Network rate (bytes/s)")
    axis.set_title("Network RX/TX rate")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()
    destination = output_dir / "network_rx_tx_bytes_per_sec.png"
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    written.append(destination)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    args = parser.parse_args(argv)
    for path in create_plots(args.input, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
