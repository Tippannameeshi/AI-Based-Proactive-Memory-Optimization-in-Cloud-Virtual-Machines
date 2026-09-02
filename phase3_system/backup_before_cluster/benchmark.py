"""
Benchmark harness: compares three memory-management modes under identical
synthetic load, and writes results to benchmark_results.csv.

Modes:
  none      - no management at all (baseline)
  reactive  - only resize AFTER memory pressure is already detected
              (simple threshold on 'unused' ratio, no prediction)
  proactive - your trained model via the full orchestrator pipeline

Usage:
    python3 benchmark.py --domain testvm --modes none reactive proactive --duration 120

Requires stress-ng installed INSIDE the guest VM (or run remotely via SSH —
see run_stress_load() below, adjust to your access method).
"""

import argparse
import csv
import subprocess
import time

import config
from collector import MemoryStatsCollector
from features import build_feature_vector
from model_wrapper import ModelWrapper
from decision_engine import DecisionEngine, ACTION_NONE
from actuator import Actuator


REACTIVE_UNUSED_THRESHOLD_KIB = 200 * 1024  # trigger reactive increase below this


def run_stress_load(vm_ssh_host, duration_seconds, vm_bytes_pct="80%"):
    """
    Launches stress-ng inside the guest over SSH. Adjust vm_ssh_host / SSH key
    setup to match your environment. Runs in the background (non-blocking).
    """
    cmd = [
        "ssh", vm_ssh_host,
        f"stress-ng --vm 2 --vm-bytes {vm_bytes_pct} --timeout {duration_seconds}s"
    ]
    print(f"[benchmark] Launching load: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def benchmark_mode(domain_name, mode, duration_seconds, vm_ssh_host, poll_interval=5):
    collector = MemoryStatsCollector(domain_name)
    actuator = Actuator(collector.dom, dry_run=False)

    model = ModelWrapper() if mode == "proactive" else None
    engine = DecisionEngine() if mode == "proactive" else None

    records = []
    stress_proc = run_stress_load(vm_ssh_host, duration_seconds)

    start = time.time()
    while time.time() - start < duration_seconds + 15:  # extra tail to observe recovery
        sample = collector.poll()
        current_mem = sample["_current_memory_kib"]
        unused = sample.get("unused", 0)

        action_taken = ACTION_NONE

        if mode == "reactive":
            if unused < REACTIVE_UNUSED_THRESHOLD_KIB:
                success, new_mem, _ = actuator.apply(
                    "increase", config.INCREMENT_KIB, current_mem)
                if success:
                    action_taken = "increase"

        elif mode == "proactive" and collector.is_ready():
            fvec = build_feature_vector(collector.get_history())
            pred_class, proba = model.predict(fvec)
            action, reason, delta = engine.decide(pred_class, proba, current_mem)
            if action != ACTION_NONE:
                success, new_mem, _ = actuator.apply(action, delta, current_mem)
                if success:
                    action_taken = action

        records.append({
            "mode": mode,
            "t": time.time() - start,
            "unused_kib": unused,
            "swap_in": sample.get("swap_in", 0),
            "swap_out": sample.get("swap_out", 0),
            "major_fault": sample.get("major_fault", 0),
            "current_memory_kib": current_mem,
            "action": action_taken,
        })

        time.sleep(poll_interval)

    stress_proc.wait()
    collector.close()
    return records


def summarize(records):
    """Compute headline metrics: peak swap activity, total faults, recovery time."""
    if not records:
        return {}
    total_swap_in = records[-1]["swap_in"] - records[0]["swap_in"]
    total_swap_out = records[-1]["swap_out"] - records[0]["swap_out"]
    total_faults = records[-1]["major_fault"] - records[0]["major_fault"]
    min_unused = min(r["unused_kib"] for r in records)

    # crude "recovery time": first point after peak load where unused returns
    # above 80% of its starting value
    baseline_unused = records[0]["unused_kib"]
    recovery_t = None
    peak_idx = min(range(len(records)), key=lambda i: records[i]["unused_kib"])
    for r in records[peak_idx:]:
        if r["unused_kib"] >= 0.8 * baseline_unused:
            recovery_t = r["t"]
            break

    return {
        "total_swap_in_kib": total_swap_in,
        "total_swap_out_kib": total_swap_out,
        "total_major_faults": total_faults,
        "min_unused_kib": min_unused,
        "recovery_time_s": recovery_t,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--modes", nargs="+", default=["none", "reactive", "proactive"])
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--vm-ssh-host", default="testvm",
                         help="SSH alias/host for the guest, e.g. set up in ~/.ssh/config")
    args = parser.parse_args()

    all_records = []
    summaries = {}

    for mode in args.modes:
        print(f"\n=== Running benchmark mode: {mode} ===")
        recs = benchmark_mode(args.domain, mode, args.duration, args.vm_ssh_host)
        all_records.extend(recs)
        summaries[mode] = summarize(recs)
        print(f"[{mode}] summary: {summaries[mode]}")
        time.sleep(10)  # cool-down between modes

    # Write raw time series
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    # Write summary table
    with open("benchmark_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "total_swap_in_kib", "total_swap_out_kib",
                          "total_major_faults", "min_unused_kib", "recovery_time_s"])
        for mode, s in summaries.items():
            writer.writerow([mode, s.get("total_swap_in_kib"), s.get("total_swap_out_kib"),
                              s.get("total_major_faults"), s.get("min_unused_kib"),
                              s.get("recovery_time_s")])

    print("\nWrote benchmark_results.csv and benchmark_summary.csv")
    print("Use benchmark_summary.csv directly as your Phase 4 comparison table/chart.")
