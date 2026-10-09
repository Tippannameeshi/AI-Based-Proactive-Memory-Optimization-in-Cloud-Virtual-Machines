#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import base64
import json
import subprocess
import time
import pandas as pd
from collections import deque

import joblib
import numpy as np
import tensorflow as tf
import libvirt


# ============================================================
# Configuration
# ============================================================

LIBVIRT_URI = "qemu:///system"

VM_NAMES = [
    "testvm1",
    "testvm2",
    "testvm3",
]

MODEL_PATH = "phase3_system/artifacts/best_proactive_memory_lstm.keras"
FEATURE_SCALER_PATH = "phase3_system/artifacts/feature_scaler.pkl"
TARGET_SCALER_PATH = "phase3_system/artifacts/target_scaler.pkl"
CONFIG_PATH = "phase3_system/artifacts/model_config.json"

POLL_INTERVAL_SECONDS = 5
SEQUENCE_LENGTH = 6

# Safety/controller values from the trained configuration.
SAFETY_MARGIN_MIB = 250.0
ALLOCATION_STEP_MIB = 64
MIN_MEMORY_MIB = 512
MAX_MEMORY_MIB = 4096


# ============================================================
# Load trained artifacts
# ============================================================

print("=" * 70)
print("LIVE LSTM PROACTIVE MEMORY INFERENCE - DRY RUN")
print("=" * 70)
print("WARNING: DRY RUN ONLY - NO VM MEMORY WILL BE CHANGED")
print()

with open(CONFIG_PATH, "r") as f:
    model_config = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

print("Model loaded:")
print(f"  Input shape : {model.input_shape}")
print(f"  Output shape: {model.output_shape}")
print(f"  Features    : {model_config['features']}")
print(f"  Sequence    : {model_config['sequence_length']}")
print(f"  Horizon     : {model_config['forecast_horizon']} samples")
print()


# ============================================================
# Guest memory via QEMU Guest Agent
# ============================================================

def get_guest_memory_used_mib(domain):
    command = {
        "execute": "guest-exec",
        "arguments": {
            "path": "/bin/sh",
            "arg": [
                "-c",
                "awk '/^MemTotal:|^MemAvailable:/ {print}' /proc/meminfo"
            ],
            "capture-output": True,
        },
    }

    result = subprocess.run(
        [
            "virsh",
            "-c",
            LIBVIRT_URI,
            "qemu-agent-command",
            domain.name(),
            json.dumps(command),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    response = json.loads(result.stdout)
    pid = response["return"]["pid"]

    while True:
        status_command = {
            "execute": "guest-exec-status",
            "arguments": {
                "pid": pid
            },
        }

        status_result = subprocess.run(
            [
                "virsh",
                "-c",
                LIBVIRT_URI,
                "qemu-agent-command",
                domain.name(),
                json.dumps(status_command),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        status = json.loads(status_result.stdout)["return"]

        if status.get("exited"):
            if status.get("exitcode") != 0:
                raise RuntimeError(
                    f"Guest memory command failed with exit code "
                    f"{status.get('exitcode')}"
                )

            encoded = status.get("out-data", "")
            output = base64.b64decode(encoded).decode("utf-8")

            mem_total_kb = None
            mem_available_kb = None

            for line in output.splitlines():
                parts = line.split()

                if len(parts) >= 2 and parts[0] == "MemTotal:":
                    mem_total_kb = float(parts[1])

                elif len(parts) >= 2 and parts[0] == "MemAvailable:":
                    mem_available_kb = float(parts[1])

            if mem_total_kb is None or mem_available_kb is None:
                raise RuntimeError(
                    f"Could not parse MemTotal/MemAvailable: {output!r}"
                )

            return (mem_total_kb - mem_available_kb) / 1024.0

        time.sleep(0.05)


# ============================================================
# CPU utilization
# ============================================================

def get_cpu_snapshot(domain):
    info = domain.info()

    # libvirt domain.info():
    # info[2] = current vCPU count
    # info[4] = cumulative CPU time in nanoseconds
    vcpus = int(info[3])
    cpu_time_ns = int(info[4])

    return cpu_time_ns, vcpus


def calculate_cpu_utilization(previous, current, interval_seconds):
    previous_cpu_ns, previous_vcpus = previous
    current_cpu_ns, current_vcpus = current

    vcpus = current_vcpus

    if vcpus <= 0:
        raise RuntimeError("VM reports zero vCPUs")

    delta_ns = current_cpu_ns - previous_cpu_ns

    if delta_ns < 0:
        return 0.0

    utilization = (
        delta_ns
        / (interval_seconds * vcpus * 1_000_000_000)
        * 100.0
    )

    return float(np.clip(utilization, 0.0, 100.0))


# ============================================================
# Allocation recommendation
# ============================================================

def recommended_allocation(predicted_memory_mib):
    raw = predicted_memory_mib + SAFETY_MARGIN_MIB

    rounded = (
        np.ceil(raw / ALLOCATION_STEP_MIB)
        * ALLOCATION_STEP_MIB
    )

    allocation = np.clip(
        rounded,
        MIN_MEMORY_MIB,
        MAX_MEMORY_MIB,
    )

    return float(allocation)


# ============================================================
# Main live loop
# ============================================================

def main():
    conn = libvirt.open(LIBVIRT_URI)

    if conn is None:
        raise RuntimeError("Could not connect to libvirt")

    domains = {}

    for vm_name in VM_NAMES:
        domain = conn.lookupByName(vm_name)

        if not domain.isActive():
            raise RuntimeError(f"{vm_name} is not running")

        domains[vm_name] = domain

    histories = {
        vm_name: deque(maxlen=SEQUENCE_LENGTH)
        for vm_name in VM_NAMES
    }

    previous_cpu = {}

    print("Connected to all VMs:")
    for vm_name in VM_NAMES:
        domain = domains[vm_name]

        cpu_snapshot = get_cpu_snapshot(domain)
        previous_cpu[vm_name] = cpu_snapshot

        print(
            f"  {vm_name}: "
            f"{cpu_snapshot[1]} vCPUs, QGA available"
        )

    print()
    print(
        f"Collecting {SEQUENCE_LENGTH} samples per VM "
        f"before first prediction..."
    )
    print(
        f"Sampling interval: {POLL_INTERVAL_SECONDS} seconds"
    )
    print()

    sample_number = 0

    try:
        while True:
            sample_number += 1

            print("-" * 70)
            print(
                f"Sample {sample_number} | "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            current_cpu = {}

            # ------------------------------------------------
            # Collect CPU snapshots first.
            # ------------------------------------------------
            for vm_name in VM_NAMES:
                current_cpu[vm_name] = get_cpu_snapshot(
                    domains[vm_name]
                )

            # ------------------------------------------------
            # Collect guest memory and calculate CPU.
            # ------------------------------------------------
            for vm_name in VM_NAMES:
                domain = domains[vm_name]

                memory_used_mib = get_guest_memory_used_mib(
                    domain
                )

                cpu_utilization_pct = calculate_cpu_utilization(
                    previous_cpu[vm_name],
                    current_cpu[vm_name],
                    POLL_INTERVAL_SECONDS,
                )

                previous_cpu[vm_name] = current_cpu[vm_name]

                sample = np.array(
                    [
                        memory_used_mib,
                        cpu_utilization_pct,
                    ],
                    dtype=np.float32,
                )

                histories[vm_name].append(sample)

                print(
                    f"{vm_name}: "
                    f"memory={memory_used_mib:8.2f} MiB | "
                    f"CPU={cpu_utilization_pct:6.2f}% | "
                    f"history={len(histories[vm_name])}/{SEQUENCE_LENGTH}"
                )

            # ------------------------------------------------
            # Predict when a full sequence exists.
            # ------------------------------------------------
            print()

            for vm_name in VM_NAMES:
                history = histories[vm_name]

                if len(history) < SEQUENCE_LENGTH:
                    print(
                        f"{vm_name}: waiting for "
                        f"{SEQUENCE_LENGTH - len(history)} more sample(s)"
                    )
                    continue

                sequence = np.array(
                    history,
                    dtype=np.float32,
                )

                sequence_df = pd.DataFrame(
                    sequence,
                    columns=[
                        "guest_memory_used_mib",
                        "cpu_utilization_pct",
                    ],
                )

                scaled_sequence = feature_scaler.transform(
                    sequence_df
                )

                model_input = scaled_sequence.reshape(
                    1,
                    SEQUENCE_LENGTH,
                    2,
                )

                scaled_prediction = model.predict(
                    model_input,
                    verbose=0,
                )

                predicted_memory_mib = float(
                    target_scaler.inverse_transform(
                        scaled_prediction
                    )[0, 0]
                )

                allocation = recommended_allocation(
                    predicted_memory_mib
                )

                print(
                    f"{vm_name}: "
                    f"predicted_future_memory="
                    f"{predicted_memory_mib:.2f} MiB | "
                    f"safety_margin={SAFETY_MARGIN_MIB:.0f} MiB | "
                    f"recommended_allocation="
                    f"{allocation:.0f} MiB"
                )

            if sample_number == 1:
                print()
                print(
                    "First sample collected. "
                    "No prediction yet; 6 samples are required."
                )

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("DRY-RUN INFERENCE STOPPED")
        print("No VM memory was changed.")
        print("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
