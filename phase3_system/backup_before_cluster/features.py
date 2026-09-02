import numpy as np

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
    "cpu_rate_of_change",
]


def _value(sample, *keys, default=0.0):
    for key in keys:
        try:
            value = sample.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return float(default)


def _series(history, *keys):
    return np.array(
        [_value(x, *keys) for x in history],
        dtype=float
    )


def build_feature_vector(history):
    """
    Build the exact 28-feature vector expected by
    models/best_model_improved.pkl.

    history is oldest -> newest.

    IMPORTANT:
    The live libvirt collector does not currently expose all
    training signals (network, power, task metadata, etc.).
    Those unavailable signals are therefore set to 0 for now.
    This gets the Phase-3 inference pipeline operational.
    """

    if not history:
        raise ValueError("History is empty")

    latest = history[-1]

    # Memory signal available from libvirt.
    memory = _series(
        history,
        "current_memory_kib",
        "memory",
        "memory_usage",
        "actual"
    )

    # CPU is not currently collected by collector.py.
    cpu = _series(history, "cpu_usage", "cpu")

    # Current time features.
    import time
    tm = time.localtime()

    hour = tm.tm_hour
    dayofweek = tm.tm_wday
    dayofmonth = tm.tm_mday
    month = tm.tm_mon
    is_weekend = 1 if dayofweek >= 5 else 0

    # Rolling helpers.
    mem5 = memory[-5:]
    mem10 = memory[-10:]

    mem_roll_mean_5 = np.mean(mem5)
    mem_roll_mean_10 = np.mean(mem10)
    mem_roll_std_5 = np.std(mem5)
    mem_roll_max_5 = np.max(mem5)

    cpu5 = cpu[-5:]

    cpu_roll_mean_5 = np.mean(cpu5)
    cpu_roll_std_5 = np.std(cpu5)

    def lag(arr, n):
        if len(arr) > n:
            return arr[-1 - n]
        return arr[0]

    memory_lag_1 = lag(memory, 1)
    memory_lag_2 = lag(memory, 2)
    memory_lag_3 = lag(memory, 3)
    memory_lag_5 = lag(memory, 5)

    cpu_lag_1 = lag(cpu, 1)
    cpu_lag_2 = lag(cpu, 2)
    cpu_lag_3 = lag(cpu, 3)

    if len(cpu) >= 2:
        cpu_rate_of_change = cpu[-1] - cpu[-2]
    else:
        cpu_rate_of_change = 0.0

    # These signals are not currently provided by collector.py.
    cpu_usage = _value(latest, "cpu_usage", "cpu", default=0.0)
    network_traffic = _value(
        latest, "network_traffic", "network", default=0.0
    )
    power_consumption = _value(
        latest, "power_consumption", "power", default=0.0
    )
    num_executed_instructions = _value(
        latest, "num_executed_instructions", default=0.0
    )
    execution_time = _value(
        latest, "execution_time", default=0.0
    )
    energy_efficiency = _value(
        latest, "energy_efficiency", default=0.0
    )

    task_type_encoded = _value(
        latest, "task_type_encoded", default=0.0
    )
    task_priority_encoded = _value(
        latest, "task_priority_encoded", default=0.0
    )
    task_status_encoded = _value(
        latest, "task_status_encoded", default=0.0
    )

    feature_vector = np.array([
        cpu_usage,
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
        mem_roll_mean_5,
        mem_roll_mean_10,
        mem_roll_std_5,
        mem_roll_max_5,
        cpu_roll_mean_5,
        cpu_roll_std_5,
        memory_lag_1,
        memory_lag_2,
        memory_lag_3,
        memory_lag_5,
        cpu_lag_1,
        cpu_lag_2,
        cpu_lag_3,
        cpu_rate_of_change,
    ], dtype=float)

    assert len(feature_vector) == 28, (
        f"Expected 28 features, got {len(feature_vector)}"
    )

    return feature_vector


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
    "cpu_rate_of_change",
]
