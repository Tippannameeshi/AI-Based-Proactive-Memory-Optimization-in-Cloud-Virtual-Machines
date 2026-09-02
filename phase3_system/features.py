"""
Feature engineering for Phase 3 live VM monitoring.

Builds the exact 28-feature vector expected by xgb_model.pkl.
"""

import numpy as np
from datetime import datetime


def _safe_series(history, key):
    """Extract numeric values safely from history."""
    return np.array(
        [float(sample.get(key, 0.0) or 0.0) for sample in history],
        dtype=float
    )


def _lag(series, n):
    """Return nth previous value safely."""
    if len(series) > n:
        return float(series[-(n + 1)])
    elif len(series) > 0:
        return float(series[0])
    return 0.0


def _calculate_memory_usage(history):
    """
    Calculate VM memory usage percentage robustly.

    libvirt memoryStats values can sometimes be inconsistent with
    dom.info()[2] depending on balloon driver / guest reporting.

    Preferred calculation:
        usage = (actual - available) / actual * 100

    Fallback:
        usage = (actual - unused) / actual * 100

    Values are always clamped to 0-100.
    """

    memory_usage_values = []

    for sample in history:

        # Use actual memory reported by balloon stats if available.
        actual = float(
            sample.get(
                "actual",
                sample.get("_current_memory_kib", 0.0)
            ) or 0.0
        )

        available = sample.get("available")
        unused = sample.get("unused")

        if actual <= 0:
            memory_usage_values.append(0.0)
            continue

        # Preferred: available memory
        if available is not None and float(available) >= 0:
            usage = (
                (actual - float(available))
                / actual
            ) * 100.0

        # Fallback: unused memory
        elif unused is not None and float(unused) >= 0:
            usage = (
                (actual - float(unused))
                / actual
            ) * 100.0

        else:
            usage = float(sample.get("memory_usage", 0.0))

        # Guard against invalid libvirt values
        usage = max(0.0, min(100.0, usage))

        memory_usage_values.append(usage)

    return np.array(memory_usage_values, dtype=float)


def build_feature_vector(history):
    """
    Build the exact 28 features expected by the trained XGBoost model.

    history: list of VM metric dictionaries
    returns: numpy array of shape (28,)
    """

    if not history:
        raise ValueError("History is empty")

    now = datetime.now()

    # ==========================================
    # LIVE CPU METRICS
    # ==========================================

    cpu = _safe_series(history, "cpu_usage")

    # ==========================================
    # MEMORY USAGE
    # ==========================================

    memory = _calculate_memory_usage(history)

    # ==========================================
    # CURRENT VALUES
    # ==========================================

    latest_cpu = float(cpu[-1]) if len(cpu) else 0.0

    # ==========================================
    # LIVE / DEFAULT SIGNALS
    # ==========================================

    latest_sample = history[-1]

    network_traffic = float(
        latest_sample.get("network_traffic", 0.0) or 0.0
    )

    power_consumption = float(
        latest_sample.get(
            "power_consumption",
            latest_cpu * 0.5
        ) or 0.0
    )

    num_executed_instructions = float(
        latest_sample.get(
            "num_executed_instructions",
            latest_cpu * 1000.0
        ) or 0.0
    )

    execution_time = float(
        latest_sample.get(
            "execution_time",
            0.0
        ) or 0.0
    )

    energy_efficiency = (
        latest_cpu / max(power_consumption, 1.0)
    )

    # ==========================================
    # TIME FEATURES
    # ==========================================

    hour = float(now.hour)
    dayofweek = float(now.weekday())
    dayofmonth = float(now.day)
    month = float(now.month)
    is_weekend = float(int(dayofweek >= 5))

    # ==========================================
    # TASK FEATURES
    # ==========================================

    task_type_encoded = float(
        latest_sample.get(
            "task_type_encoded",
            0
        ) or 0
    )

    task_priority_encoded = float(
        latest_sample.get(
            "task_priority_encoded",
            0
        ) or 0
    )

    task_status_encoded = float(
        latest_sample.get(
            "task_status_encoded",
            0
        ) or 0
    )

    # ==========================================
    # MEMORY ROLLING FEATURES
    # ==========================================

    last_5_memory = (
        memory[-5:]
        if len(memory)
        else np.array([0.0])
    )

    last_10_memory = (
        memory[-10:]
        if len(memory)
        else np.array([0.0])
    )

    memory_roll_mean_5 = float(
        np.mean(last_5_memory)
    )

    memory_roll_mean_10 = float(
        np.mean(last_10_memory)
    )

    memory_roll_std_5 = float(
        np.std(last_5_memory)
    )

    memory_roll_max_5 = float(
        np.max(last_5_memory)
    )

    # ==========================================
    # CPU ROLLING FEATURES
    # ==========================================

    last_5_cpu = (
        cpu[-5:]
        if len(cpu)
        else np.array([0.0])
    )

    cpu_roll_mean_5 = float(
        np.mean(last_5_cpu)
    )

    cpu_roll_std_5 = float(
        np.std(last_5_cpu)
    )

    # ==========================================
    # MEMORY LAG FEATURES
    # ==========================================

    memory_lag_1 = _lag(memory, 1)
    memory_lag_2 = _lag(memory, 2)
    memory_lag_3 = _lag(memory, 3)
    memory_lag_5 = _lag(memory, 5)

    # ==========================================
    # CPU LAG FEATURES
    # ==========================================

    cpu_lag_1 = _lag(cpu, 1)
    cpu_lag_2 = _lag(cpu, 2)
    cpu_lag_3 = _lag(cpu, 3)

    # ==========================================
    # CPU RATE OF CHANGE
    # ==========================================

    if len(cpu) >= 2:
        cpu_rate_of_change = float(
            cpu[-1] - cpu[-2]
        )
    else:
        cpu_rate_of_change = 0.0

    # ==========================================
    # EXACT 28 FEATURE ORDER
    # DO NOT CHANGE ORDER
    # ==========================================

    feature_vector = np.array([
        latest_cpu,
        network_traffic,
        power_consumption,
        num_executed_instructions,
        execution_time,
        energy_efficiency,
        hour,
        dayofweek,
        dayofmonth,
        month,
        is_weekend,
        task_type_encoded,
        task_priority_encoded,
        task_status_encoded,
        memory_roll_mean_5,
        memory_roll_mean_10,
        memory_roll_std_5,
        memory_roll_max_5,
        cpu_roll_mean_5,
        cpu_roll_std_5,
        memory_lag_1,
        memory_lag_2,
        memory_lag_3,
        memory_lag_5,
        cpu_lag_1,
        cpu_lag_2,
        cpu_lag_3,
        cpu_rate_of_change
    ], dtype=float)

    # Final safety check
    if len(feature_vector) != 28:
        raise RuntimeError(
            f"Feature vector has {len(feature_vector)} features, expected 28"
        )

    if not np.all(np.isfinite(feature_vector)):
        raise RuntimeError(
            "Feature vector contains NaN or infinite values"
        )

    return feature_vector


# ==========================================
# FEATURE NAMES
# MUST STAY IN EXACT ORDER
# ==========================================

FEATURE_NAMES = [
    "cpu_usage",
    "network_traffic",
    "power_consumption",
    "num_executed_instructions",
    "execution_time",
    "energy_efficiency",
    "hour",
    "dayofweek",
    "dayofmonth",
    "month",
    "is_weekend",
    "task_type_encoded",
    "task_priority_encoded",
    "task_status_encoded",
    "memory_roll_mean_5",
    "memory_roll_mean_10",
    "memory_roll_std_5",
    "memory_roll_max_5",
    "cpu_roll_mean_5",
    "cpu_roll_std_5",
    "memory_lag_1",
    "memory_lag_2",
    "memory_lag_3",
    "memory_lag_5",
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "cpu_rate_of_change"
]


if __name__ == "__main__":

    print("Number of features:", len(FEATURE_NAMES))
    print()

    for i, feature in enumerate(FEATURE_NAMES):
        print(f"{i}: {feature}")
