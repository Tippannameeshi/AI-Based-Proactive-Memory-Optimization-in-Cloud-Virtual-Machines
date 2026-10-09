#!/usr/bin/env python3

import argparse
import importlib
import math
import shlex
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


VM_NAMES = ("testvm1", "testvm2", "testvm3")
POLL_INTERVAL_SECONDS = 5.0
SEQUENCE_LENGTH = 6
FORECAST_HORIZON = 2
SAFETY_MARGIN_MIB = 250.0
ALLOCATION_STEP_MIB = 64
MIN_MEMORY_MIB = 512
MAX_MEMORY_MIB = 4096
MIN_GUEST_HEADROOM_MIB = 128
REAL_ACTION_COOLDOWN_SECONDS = 30.0
VERIFICATION_TIMEOUT_SECONDS = 5.0
VERIFICATION_POLL_SECONDS = 0.1
FEATURE_NAMES = ("guest_memory_used_mib", "cpu_utilization_pct")


class TelemetryBaselinePending(RuntimeError):
    pass


@dataclass(frozen=True)
class BaselineAllocation:
    memory_kib: int


@dataclass
class VMState:
    name: str
    domain: object = None
    history: deque = field(
        default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH)
    )
    previous_cpu_snapshot: Optional[tuple] = None
    previous_monotonic_timestamp: Optional[float] = None
    actuator: object = None
    last_real_action_timestamp: Optional[float] = None
    baseline: Optional[BaselineAllocation] = None


def _log(**fields):
    fields.setdefault(
        "timestamp",
        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
    message = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in fields.items()
    )
    print(message, flush=True)


def _load_inference_module():
    phase3_dir = Path(__file__).resolve().parent
    repo_root = phase3_dir.parent
    if str(phase3_dir) not in sys.path:
        sys.path.insert(0, str(phase3_dir))

    previous_cwd = Path.cwd()
    try:
        # The validated inference module resolves its artifact paths from the
        # project root; restore the caller's working directory after loading.
        import os

        os.chdir(repo_root)
        from contextlib import redirect_stdout
        from io import StringIO

        with redirect_stdout(StringIO()):
            return importlib.import_module("live_lstm_inference")
    finally:
        os.chdir(previous_cwd)


def _validate_inference_contract(inference):
    model_config = inference.model_config
    if model_config.get("features") != list(FEATURE_NAMES):
        raise RuntimeError("LSTM feature order does not match the live contract")
    if model_config.get("sequence_length") != SEQUENCE_LENGTH:
        raise RuntimeError("LSTM sequence length does not match the live contract")
    if model_config.get("forecast_horizon") != FORECAST_HORIZON:
        raise RuntimeError("LSTM forecast horizon does not match the live contract")
    if inference.SAFETY_MARGIN_MIB != SAFETY_MARGIN_MIB:
        raise RuntimeError("LSTM safety margin differs from the approved value")
    if inference.ALLOCATION_STEP_MIB != ALLOCATION_STEP_MIB:
        raise RuntimeError("LSTM allocation step differs from the approved value")
    if inference.MIN_MEMORY_MIB != MIN_MEMORY_MIB:
        raise RuntimeError("LSTM minimum allocation differs from the approved value")
    if inference.MAX_MEMORY_MIB != MAX_MEMORY_MIB:
        raise RuntimeError("LSTM maximum allocation differs from the approved value")

    input_shape = tuple(inference.model.input_shape)
    output_shape = tuple(inference.model.output_shape)
    if (
        len(input_shape) != 3
        or input_shape[1:] != (SEQUENCE_LENGTH, len(FEATURE_NAMES))
    ):
        raise RuntimeError(f"Unexpected LSTM input shape: {input_shape}")
    if len(output_shape) != 2 or output_shape[1] != 1:
        raise RuntimeError(f"Unexpected LSTM output shape: {output_shape}")


def _finite_number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _read_domain_memory(state):
    if state.domain is None or not state.domain.isActive():
        raise RuntimeError("domain is not active")

    info = state.domain.info()
    if len(info) < 3:
        raise RuntimeError("libvirt returned incomplete domain information")

    current_kib = _finite_number(info[2])
    maximum_kib = _finite_number(info[1])
    if current_kib is None or maximum_kib is None:
        raise RuntimeError("libvirt returned invalid memory allocation values")
    if current_kib <= 0 or maximum_kib <= 0:
        raise RuntimeError("libvirt returned non-positive memory allocation values")
    if not current_kib.is_integer() or not maximum_kib.is_integer():
        raise RuntimeError("libvirt returned non-integer memory allocation values")

    return int(current_kib), int(maximum_kib)


def _collect_sample(state, inference):
    if state.domain is None or not state.domain.isActive():
        state.history.clear()
        state.previous_cpu_snapshot = None
        state.previous_monotonic_timestamp = None
        raise RuntimeError("domain is not active")

    current_cpu = inference.get_cpu_snapshot(state.domain)
    sample_timestamp = time.monotonic()
    if len(current_cpu) != 2 or int(current_cpu[1]) <= 0:
        state.history.clear()
        state.previous_cpu_snapshot = current_cpu
        state.previous_monotonic_timestamp = sample_timestamp
        raise RuntimeError("invalid CPU snapshot")

    if (
        state.previous_cpu_snapshot is None
        or state.previous_monotonic_timestamp is None
    ):
        state.previous_cpu_snapshot = current_cpu
        state.previous_monotonic_timestamp = sample_timestamp
        raise TelemetryBaselinePending(
            "CPU baseline initialized; waiting for next sample"
        )

    elapsed_seconds = sample_timestamp - state.previous_monotonic_timestamp
    previous_cpu = state.previous_cpu_snapshot
    state.previous_cpu_snapshot = current_cpu
    state.previous_monotonic_timestamp = sample_timestamp

    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        state.history.clear()
        raise RuntimeError("invalid elapsed time for CPU utilization")
    if int(current_cpu[0]) < int(previous_cpu[0]):
        state.history.clear()
        raise RuntimeError("CPU time counter moved backwards")

    cpu_pct = inference.calculate_cpu_utilization(
        previous_cpu,
        current_cpu,
        elapsed_seconds,
    )
    memory_mib = inference.get_guest_memory_used_mib(state.domain)

    memory_mib = _finite_number(memory_mib)
    cpu_pct = _finite_number(cpu_pct)
    if memory_mib is None or memory_mib < 0:
        state.history.clear()
        raise RuntimeError("guest memory telemetry is invalid")
    if cpu_pct is None or not 0 <= cpu_pct <= 100:
        state.history.clear()
        raise RuntimeError("CPU utilization telemetry is invalid")

    sample = [memory_mib, cpu_pct]
    state.history.append(sample)
    return sample


def _predict_memory(state, inference):
    import numpy as np
    import pandas as pd

    if len(state.history) != SEQUENCE_LENGTH:
        raise RuntimeError("prediction requested before history is full")

    sequence = np.asarray(state.history, dtype=np.float32)
    if sequence.shape != (SEQUENCE_LENGTH, len(FEATURE_NAMES)):
        raise RuntimeError(f"invalid feature sequence shape: {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise RuntimeError("feature sequence contains NaN or infinity")

    sequence_frame = pd.DataFrame(sequence, columns=list(FEATURE_NAMES))
    scaled_sequence = inference.feature_scaler.transform(sequence_frame)
    scaled_sequence = np.asarray(scaled_sequence)
    if (
        scaled_sequence.shape != sequence.shape
        or not np.isfinite(scaled_sequence).all()
    ):
        raise RuntimeError("feature scaler returned invalid values or shape")

    model_input = scaled_sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURE_NAMES))
    scaled_prediction = np.asarray(
        inference.model.predict(model_input, verbose=0)
    )
    if (
        scaled_prediction.shape != (1, 1)
        or not np.isfinite(scaled_prediction).all()
    ):
        raise RuntimeError(
            f"model returned invalid prediction shape or values: "
            f"{scaled_prediction.shape}"
        )

    prediction = np.asarray(
        inference.target_scaler.inverse_transform(scaled_prediction)
    )
    if prediction.shape != (1, 1) or not np.isfinite(prediction).all():
        raise RuntimeError("target scaler returned an invalid prediction")

    predicted_mib = float(prediction[0, 0])
    if predicted_mib < 0:
        raise RuntimeError("model predicted negative memory usage")

    allocation_mib = inference.recommended_allocation(predicted_mib)
    allocation_mib = _finite_number(allocation_mib)
    if allocation_mib is None or not allocation_mib.is_integer():
        raise RuntimeError("recommended allocation is not an integer MiB value")
    target_mib = int(allocation_mib)
    if target_mib % ALLOCATION_STEP_MIB != 0:
        raise RuntimeError("recommended allocation is not on a 64 MiB boundary")
    if not MIN_MEMORY_MIB <= target_mib <= MAX_MEMORY_MIB:
        raise RuntimeError("recommended allocation is outside configured bounds")

    return predicted_mib, target_mib


def _make_plan(state, target_mib, cooldown_seconds):
    current_kib, maximum_kib = _read_domain_memory(state)
    target_kib = target_mib * 1024

    if target_kib > maximum_kib:
        return {
            "current_kib": current_kib,
            "maximum_kib": maximum_kib,
            "target_kib": target_kib,
            "action": "none",
            "delta_kib": 0,
            "cooldown": "allowed",
            "result": "blocked",
            "error": "target exceeds domain maximum",
        }

    delta_kib = target_kib - current_kib
    action = "increase" if delta_kib > 0 else "decrease" if delta_kib < 0 else "none"
    if (
        (action == "increase" and delta_kib <= 0)
        or (action == "decrease" and delta_kib >= 0)
        or (action == "none" and delta_kib != 0)
    ):
        return {
            "current_kib": current_kib,
            "maximum_kib": maximum_kib,
            "target_kib": target_kib,
            "action": action,
            "delta_kib": delta_kib,
            "cooldown": "allowed",
            "result": "blocked",
            "error": "action and delta sign do not agree",
        }

    cooldown = "allowed"
    if delta_kib and state.last_real_action_timestamp is not None:
        elapsed = time.monotonic() - state.last_real_action_timestamp
        if elapsed < cooldown_seconds:
            cooldown = "blocked"

    result = "would_skip" if action == "none" else "would_apply"
    if action != "none" and cooldown == "blocked":
        result = "blocked"

    return {
        "current_kib": current_kib,
        "maximum_kib": maximum_kib,
        "target_kib": target_kib,
        "action": action,
        "delta_kib": delta_kib,
        "cooldown": cooldown,
        "result": result,
        "error": None,
    }


def _verify_allocation(state, target_kib):
    deadline = time.monotonic() + VERIFICATION_TIMEOUT_SECONDS
    observed_kib = None
    had_valid_observation = False
    last_error = None

    while True:
        try:
            observed_kib, maximum_kib = _read_domain_memory(state)
            had_valid_observation = True
            if observed_kib > maximum_kib:
                return (
                    "mismatch",
                    observed_kib,
                    "observed allocation exceeds domain maximum",
                )
            if observed_kib == target_kib:
                return "verified", observed_kib, None
        except Exception as exc:
            last_error = exc

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(VERIFICATION_POLL_SECONDS, remaining))

    if had_valid_observation:
        return "timeout", observed_kib, None
    return "error", observed_kib, str(last_error or "no valid readback")


def _execute_live_action(state, plan, actuator_class, inference):
    current_kib, maximum_kib = _read_domain_memory(state)
    target_kib = plan["target_kib"]
    if target_kib > maximum_kib:
        _log(
            vm_id=state.name,
            event="action",
            mode="live",
            requested_kib=target_kib,
            observed_kib=current_kib,
            verification="error",
            result="blocked",
            error="target exceeds freshly read domain maximum",
        )
        return

    delta_kib = target_kib - current_kib
    action = "increase" if delta_kib > 0 else "decrease" if delta_kib < 0 else "none"
    if action == "none":
        _log(
            vm_id=state.name,
            event="action",
            mode="live",
            requested_kib=target_kib,
            observed_kib=current_kib,
            verification="verified",
            result="would_skip",
            error="allocation already equals target",
        )
        return

    if state.last_real_action_timestamp is not None:
        elapsed = time.monotonic() - state.last_real_action_timestamp
        if elapsed < REAL_ACTION_COOLDOWN_SECONDS:
            _log(
                vm_id=state.name,
                event="action",
                mode="live",
                requested_kib=target_kib,
                observed_kib=current_kib,
                verification="error",
                result="blocked",
                error="per-VM cooldown became active before actuation",
            )
            return

    if state.actuator is None or state.actuator.dom != state.domain:
        state.actuator = actuator_class(state.domain, dry_run=False)

    current_guest_memory_mib = _finite_number(
        inference.get_guest_memory_used_mib(state.domain)
    )
    if current_guest_memory_mib is None or current_guest_memory_mib < 0:
        _log(
            vm_id=state.name,
            event="decision",
            mode="live",
            result="blocked_invalid_guest_memory",
            current_guest_memory_mib=current_guest_memory_mib,
            target_memory_mib=target_kib / 1024,
            error="fresh guest memory telemetry is invalid",
        )
        return

    target_memory_mib = target_kib / 1024
    required_minimum_mib = (
        current_guest_memory_mib + MIN_GUEST_HEADROOM_MIB
    )
    if target_memory_mib < required_minimum_mib:
        _log(
            vm_id=state.name,
            event="decision",
            mode="live",
            result="blocked_guest_headroom",
            current_guest_memory_mib=f"{current_guest_memory_mib:.3f}",
            target_memory_mib=f"{target_memory_mib:.3f}",
            required_minimum_mib=f"{required_minimum_mib:.3f}",
        )
        return

    state.last_real_action_timestamp = time.monotonic()
    actuator_success = False
    actuator_observed_kib = None
    actuator_message = ""
    actuator_error = None
    try:
        actuator_success, actuator_observed_kib, actuator_message = (
            state.actuator.apply(action, delta_kib, current_kib)
        )
    except Exception as exc:
        actuator_error = str(exc)

    verification, observed_kib, verification_error = _verify_allocation(
        state,
        target_kib,
    )
    result = "success" if actuator_success and verification == "verified" else "error"
    _log(
        vm_id=state.name,
        event="action",
        mode="live",
        requested_kib=target_kib,
        observed_kib=observed_kib,
        actuator_observed_kib=actuator_observed_kib,
        verification=verification,
        result=result,
        actuator_success=actuator_success,
        actuator_message=actuator_message,
        error=actuator_error or verification_error or "",
    )


def _restore_vm_baseline(state, actuator_class, inference):
    baseline_kib = state.baseline.memory_kib
    current_kib, maximum_kib = _read_domain_memory(state)
    current_guest_memory_mib = _finite_number(
        inference.get_guest_memory_used_mib(state.domain)
    )
    if current_guest_memory_mib is None or current_guest_memory_mib < 0:
        _log(
            vm_id=state.name,
            event="restoration",
            baseline_kib=baseline_kib,
            current_kib=current_kib,
            requested_kib=baseline_kib,
            delta_kib=baseline_kib - current_kib,
            result="restore_blocked",
            verification="not_attempted",
            error="fresh guest memory telemetry is invalid",
        )
        return

    if current_kib == baseline_kib:
        _log(
            vm_id=state.name,
            event="restoration",
            baseline_kib=baseline_kib,
            current_kib=current_kib,
            requested_kib=baseline_kib,
            delta_kib=0,
            result="no_restoration_needed",
            verification="not_required",
        )
        return

    delta_kib = baseline_kib - current_kib
    action = "increase" if delta_kib > 0 else "decrease"
    baseline_mib = baseline_kib / 1024
    if not MIN_MEMORY_MIB * 1024 <= baseline_kib <= MAX_MEMORY_MIB * 1024:
        _log(
            vm_id=state.name,
            event="restoration",
            baseline_kib=baseline_kib,
            current_kib=current_kib,
            requested_kib=baseline_kib,
            delta_kib=delta_kib,
            result="restore_blocked",
            verification="not_attempted",
            error="baseline is outside actuator safety bounds",
        )
        return

    if baseline_kib > maximum_kib:
        _log(
            vm_id=state.name,
            event="restoration",
            baseline_kib=baseline_kib,
            current_kib=current_kib,
            requested_kib=baseline_kib,
            delta_kib=delta_kib,
            result="restore_blocked",
            verification="not_attempted",
            error="baseline exceeds current domain maximum",
        )
        return

    if delta_kib < 0:
        required_minimum_mib = (
            current_guest_memory_mib + MIN_GUEST_HEADROOM_MIB
        )
        if baseline_mib < required_minimum_mib:
            _log(
                vm_id=state.name,
                event="restoration",
                baseline_kib=baseline_kib,
                current_kib=current_kib,
                requested_kib=baseline_kib,
                delta_kib=delta_kib,
                current_guest_memory_mib=f"{current_guest_memory_mib:.3f}",
                baseline_target_mib=f"{baseline_mib:.3f}",
                required_minimum_mib=f"{required_minimum_mib:.3f}",
                result="restore_blocked",
                verification="not_attempted",
            )
            return

    if state.actuator is None or state.actuator.dom != state.domain:
        state.actuator = actuator_class(state.domain, dry_run=False)

    actuator_success = False
    actuator_observed_kib = None
    actuator_message = ""
    actuator_error = None
    try:
        actuator_success, actuator_observed_kib, actuator_message = (
            state.actuator.apply(action, delta_kib, current_kib)
        )
    except Exception as exc:
        actuator_error = str(exc)

    verification, observed_kib, verification_error = _verify_allocation(
        state,
        baseline_kib,
    )
    restored = (
        actuator_success
        and verification == "verified"
        and observed_kib == baseline_kib
    )
    _log(
        vm_id=state.name,
        event="restoration",
        baseline_kib=baseline_kib,
        current_kib=current_kib,
        requested_kib=baseline_kib,
        delta_kib=delta_kib,
        actuator_observed_kib=actuator_observed_kib,
        observed_kib=observed_kib,
        result="restored" if restored else "restore_failed",
        verification=verification,
        actuator_success=actuator_success,
        actuator_message=actuator_message,
        error=actuator_error or verification_error or "",
    )


def run(live=False, vm_name=None):
    if vm_name is not None and vm_name not in VM_NAMES:
        raise ValueError(f"unsupported VM name: {vm_name}")

    selected_vm_names = (vm_name,) if vm_name is not None else VM_NAMES

    import libvirt

    inference = _load_inference_module()
    _validate_inference_contract(inference)

    actuator_class = None
    if live:
        phase3_dir = str(Path(__file__).resolve().parent)
        if phase3_dir not in sys.path:
            sys.path.insert(0, phase3_dir)
        actuator_module = importlib.import_module("actuator")
        actuator_class = actuator_module.Actuator

    mode = "live" if live else "dry_run"
    _log(event="startup", mode=mode, vms=",".join(selected_vm_names))

    connection = libvirt.open(inference.LIBVIRT_URI)
    if connection is None:
        raise RuntimeError(f"Could not connect to libvirt at {inference.LIBVIRT_URI}")

    states = {name: VMState(name=name) for name in selected_vm_names}
    try:
        active_count = 0
        for state in states.values():
            try:
                state.domain = connection.lookupByName(state.name)
                if not state.domain.isActive():
                    raise RuntimeError("domain is not active")
                active_count += 1
                if live:
                    try:
                        baseline_kib, _ = _read_domain_memory(state)
                        state.baseline = BaselineAllocation(baseline_kib)
                        _log(
                            vm_id=state.name,
                            event="baseline_captured",
                            mode=mode,
                            baseline_kib=baseline_kib,
                        )
                    except Exception as exc:
                        _log(
                            vm_id=state.name,
                            stage="baseline_capture",
                            result="live_actions_disabled",
                            error=exc,
                        )
                state.previous_cpu_snapshot = inference.get_cpu_snapshot(state.domain)
                state.previous_monotonic_timestamp = time.monotonic()
                _log(
                    vm_id=state.name,
                    event="connected",
                    mode=mode,
                    vcpus=state.previous_cpu_snapshot[1],
                    history=f"0/{SEQUENCE_LENGTH}",
                )
            except Exception as exc:
                _log(vm_id=state.name, stage="connect", error=exc)
        if active_count == 0:
            raise RuntimeError("None of the configured VMs is active")

        _log(
            event="monitoring",
            mode=mode,
            interval_seconds=POLL_INTERVAL_SECONDS,
            sequence_length=SEQUENCE_LENGTH,
            forecast_horizon=FORECAST_HORIZON,
            cooldown_seconds=REAL_ACTION_COOLDOWN_SECONDS,
        )

        while True:
            cycle_start = time.monotonic()
            for state in states.values():
                try:
                    if state.domain is None:
                        state.domain = connection.lookupByName(state.name)

                    sample = _collect_sample(state, inference)
                    _log(
                        vm_id=state.name,
                        event="telemetry",
                        mode=mode,
                        history=f"{len(state.history)}/{SEQUENCE_LENGTH}",
                        guest_memory_used_mib=f"{sample[0]:.3f}",
                        cpu_utilization_pct=f"{sample[1]:.3f}",
                    )
                except Exception as exc:
                    if isinstance(exc, TelemetryBaselinePending):
                        _log(
                            vm_id=state.name,
                            event="warmup",
                            mode=mode,
                            history=f"{len(state.history)}/{SEQUENCE_LENGTH}",
                            result="waiting_for_cpu_baseline",
                        )
                    else:
                        state.history.clear()
                        _log(vm_id=state.name, stage="telemetry", error=exc)
                    continue

                if len(state.history) < SEQUENCE_LENGTH:
                    continue

                if live and state.baseline is None:
                    _log(
                        vm_id=state.name,
                        event="decision",
                        mode=mode,
                        history=f"{len(state.history)}/{SEQUENCE_LENGTH}",
                        result="blocked_baseline_missing",
                        error="live action disabled because baseline capture failed",
                    )
                    continue

                try:
                    predicted_mib, target_mib = _predict_memory(state, inference)
                except Exception as exc:
                    _log(
                        vm_id=state.name,
                        stage="prediction",
                        history=f"{len(state.history)}/{SEQUENCE_LENGTH}",
                        error=exc,
                    )
                    continue

                try:
                    plan = _make_plan(
                        state,
                        target_mib,
                        REAL_ACTION_COOLDOWN_SECONDS,
                    )
                except Exception as exc:
                    _log(vm_id=state.name, stage="allocation", error=exc)
                    continue

                if (
                    live
                    and plan["result"] == "would_apply"
                    and plan["cooldown"] == "allowed"
                ):
                    try:
                        # Refresh allocation immediately before deciding the real delta.
                        current_kib, maximum_kib = _read_domain_memory(state)
                        target_kib = target_mib * 1024
                        if target_kib > maximum_kib:
                            plan.update(
                                current_kib=current_kib,
                                maximum_kib=maximum_kib,
                                target_kib=target_kib,
                                action="none",
                                delta_kib=0,
                                cooldown="allowed",
                                result="blocked",
                                error="target exceeds freshly read domain maximum",
                            )
                        else:
                            delta_kib = target_kib - current_kib
                            action = (
                                "increase" if delta_kib > 0
                                else "decrease" if delta_kib < 0
                                else "none"
                            )
                            plan.update(
                                current_kib=current_kib,
                                maximum_kib=maximum_kib,
                                action=action,
                                delta_kib=delta_kib,
                                result="would_skip" if action == "none" else "would_apply",
                            )
                    except Exception as exc:
                        _log(vm_id=state.name, stage="allocation_refresh", error=exc)
                        continue

                _log(
                    vm_id=state.name,
                    event="decision",
                    mode=mode,
                    history=f"{len(state.history)}/{SEQUENCE_LENGTH}",
                    predicted_memory_mib=f"{predicted_mib:.3f}",
                    recommended_mib=target_mib,
                    current_kib=plan["current_kib"],
                    target_kib=plan["target_kib"],
                    action=plan["action"],
                    delta_kib=plan["delta_kib"],
                    cooldown=plan["cooldown"],
                    result=plan["result"],
                    error=plan["error"] or "",
                )

                if plan["result"] == "blocked" or plan["action"] == "none":
                    continue
                if not live:
                    continue
                if plan["cooldown"] == "blocked":
                    continue

                try:
                    _execute_live_action(
                        state,
                        plan,
                        actuator_class,
                        inference,
                    )
                except Exception as exc:
                    _log(
                        vm_id=state.name,
                        event="action",
                        mode="live",
                        requested_kib=plan["target_kib"],
                        observed_kib="unknown",
                        verification="error",
                        result="error",
                        error=exc,
                    )

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, POLL_INTERVAL_SECONDS - elapsed))
    except KeyboardInterrupt:
        _log(event="stopped", mode=mode, reason="keyboard_interrupt")
    finally:
        if live:
            for state in states.values():
                if state.baseline is None:
                    _log(
                        vm_id=state.name,
                        event="restoration",
                        baseline_kib="unknown",
                        current_kib="unknown",
                        requested_kib="unknown",
                        delta_kib="unknown",
                        result="restore_blocked",
                        verification="not_attempted",
                        error="no baseline was captured; live actions were disabled",
                    )
                    continue
                try:
                    _restore_vm_baseline(state, actuator_class, inference)
                except Exception as exc:
                    _log(
                        vm_id=state.name,
                        event="restoration",
                        baseline_kib=state.baseline.memory_kib,
                        current_kib="unknown",
                        requested_kib=state.baseline.memory_kib,
                        delta_kib="unknown",
                        result="restore_failed",
                        verification="error",
                        error=exc,
                    )
        connection.close()
        _log(event="shutdown", mode=mode, result="connection_closed")


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop proactive LSTM VM memory orchestrator."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable real memory actuation (default is dry-run).",
    )
    parser.add_argument(
        "--vm",
        choices=VM_NAMES,
        metavar="VM_NAME",
        help="Monitor only one configured VM (default: all three).",
    )
    args = parser.parse_args()
    run(live=args.live, vm_name=args.vm)


if __name__ == "__main__":
    main()
