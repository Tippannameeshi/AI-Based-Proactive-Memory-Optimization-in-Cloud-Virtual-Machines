"""
Polls libvirt for live VM memory and CPU statistics.
"""

import collections
import time
import libvirt

import config


class MemoryStatsCollector:

    def __init__(
        self,
        domain_name,
        uri=config.LIBVIRT_URI,
        window_size=config.HISTORY_WINDOW_SIZE
    ):
        self.conn = libvirt.open(uri)

        if self.conn is None:
            raise RuntimeError(
                f"Failed to open libvirt connection to {uri}"
            )

        self.dom = self.conn.lookupByName(domain_name)

        if self.dom is None:
            raise RuntimeError(
                f"Domain '{domain_name}' not found"
            )

        self.history = collections.deque(
            maxlen=window_size
        )

        # CPU tracking
        self._prev_cpu_time = None
        self._prev_wall_time = None


    def _get_cpu_usage(self, now):
        """
        Calculate VM CPU usage percentage.
        """

        cpu_stats = self.dom.getCPUStats(True)

        cpu_time = sum(
            stat.get("cpu_time", 0)
            for stat in cpu_stats
        )

        # First sample has no previous value
        if (
            self._prev_cpu_time is None
            or self._prev_wall_time is None
        ):
            self._prev_cpu_time = cpu_time
            self._prev_wall_time = now
            return 0.0

        cpu_delta = (
            cpu_time - self._prev_cpu_time
        )

        wall_delta = (
            now - self._prev_wall_time
        )

        self._prev_cpu_time = cpu_time
        self._prev_wall_time = now

        if wall_delta <= 0:
            return 0.0

        # cpu_time is nanoseconds
        cpu_usage = (
            cpu_delta
            / (wall_delta * 1_000_000_000)
        ) * 100.0

        return max(0.0, cpu_usage)


    def _calculate_memory_usage(self, raw, current_memory_kib):
        """
        Calculate memory usage percentage.

        IMPORTANT:
        In this VM, libvirt's 'available' and 'unused'
        values can reflect the guest's maximum memory rather
        than the current balloon allocation.

        Therefore we use RSS when available:

            memory_usage = rss / current_memory * 100
        """

        rss = raw.get("rss")

        if (
            rss is not None
            and current_memory_kib > 0
        ):
            usage = (
                float(rss)
                / float(current_memory_kib)
            ) * 100.0

            return max(
                0.0,
                min(100.0, usage)
            )

        # Fallback if RSS unavailable
        actual = raw.get(
            "actual",
            current_memory_kib
        )

        if actual <= 0:
            return 0.0

        unused = raw.get("unused")

        if unused is not None:
            usage = (
                (actual - float(unused))
                / float(actual)
            ) * 100.0

            return max(
                0.0,
                min(100.0, usage)
            )

        return 0.0


    def poll(self):
        """
        Collect one live VM statistics sample.
        """

        # Raw libvirt memory statistics
        raw = self.dom.memoryStats()

        now = time.time()

        raw["_timestamp"] = now

        # ----------------------------------
        # DOMAIN MEMORY INFORMATION
        # ----------------------------------

        info = self.dom.info()

        # info format:
        # [state, maxMem, memory, nrVirtCpu, cpuTime]

        raw["_max_memory_kib"] = int(info[1])

        raw["_current_memory_kib"] = int(info[2])

        # ----------------------------------
        # CPU USAGE
        # ----------------------------------

        raw["cpu_usage"] = (
            self._get_cpu_usage(now)
        )

        # ----------------------------------
        # MEMORY USAGE
        # ----------------------------------

        raw["memory_usage"] = (
            self._calculate_memory_usage(
                raw,
                raw["_current_memory_kib"]
            )
        )

        # ----------------------------------
        # DEFAULT / OPTIONAL METRICS
        # ----------------------------------

        raw["network_traffic"] = float(
            raw.get(
                "network_traffic",
                0.0
            )
        )

        # Use CPU-based fallback rather than
        # permanently zero values
        raw["power_consumption"] = float(
            raw.get(
                "power_consumption",
                raw["cpu_usage"] * 0.5
            )
        )

        raw["num_executed_instructions"] = float(
            raw.get(
                "num_executed_instructions",
                raw["cpu_usage"] * 1000.0
            )
        )

        raw["execution_time"] = float(
            raw.get(
                "execution_time",
                0.0
            )
        )

        # ----------------------------------
        # TASK FEATURES
        # ----------------------------------

        raw["task_type_encoded"] = int(
            raw.get(
                "task_type_encoded",
                0
            )
        )

        raw["task_priority_encoded"] = int(
            raw.get(
                "task_priority_encoded",
                0
            )
        )

        raw["task_status_encoded"] = int(
            raw.get(
                "task_status_encoded",
                0
            )
        )

        # ----------------------------------
        # STORE HISTORY
        # ----------------------------------

        self.history.append(raw)

        return raw


    def get_history(self):
        """Return collected history."""
        return list(self.history)


    def is_ready(self):
        """Check whether enough samples exist."""
        return (
            len(self.history)
            == self.history.maxlen
        )


    def close(self):
        """Close libvirt connection."""
        if self.conn is not None:
            self.conn.close()


if __name__ == "__main__":

    import sys
    import json

    domain = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "testvm"
    )

    c = MemoryStatsCollector(domain)

    try:

        for i in range(10):

            sample = c.poll()

            print(
                f"\n========== SAMPLE {i + 1} =========="
            )

            print(
                json.dumps(
                    sample,
                    indent=2
                )
            )

            time.sleep(
                config.POLL_INTERVAL_SECONDS
            )

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        c.close()
