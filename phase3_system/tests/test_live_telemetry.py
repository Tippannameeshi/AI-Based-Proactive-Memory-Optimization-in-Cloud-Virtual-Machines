import unittest

from phase3_system.live_telemetry_collector import (
    TelemetryError,
    counter_rate,
    normalized_cpu_percent,
    parse_meminfo,
)
from phase3_system.workload_generator import (
    MAX_TOTAL_MEMORY_MIB,
    validate_workload_request,
)
from phase3_system.collect_live_dataset import (
    WORKLOAD_PROFILES,
    _next_vm_group,
    _workload_variant,
)
from collections import Counter


class TelemetryCalculationsTest(unittest.TestCase):
    def test_parse_meminfo_requires_and_reads_guest_memory_and_swap(self):
        parsed = parse_meminfo(
            "MemTotal:       4096000 kB\n"
            "MemAvailable:   2048000 kB\n"
            "SwapTotal:       512000 kB\n"
            "SwapFree:        256000 kB\n"
        )
        self.assertEqual(parsed["MemTotal"], 4096000)
        self.assertEqual(parsed["MemAvailable"], 2048000)

    def test_parse_meminfo_rejects_missing_required_fields(self):
        with self.assertRaises(TelemetryError):
            parse_meminfo("MemTotal: 1024 kB\n")

    def test_counter_rate_uses_observed_elapsed_time(self):
        self.assertEqual(counter_rate(100, 350, 5.5), 250 / 5.5)
        self.assertIsNone(counter_rate(350, 100, 5.0))
        self.assertIsNone(counter_rate(100, 200, None))

    def test_cpu_utilization_is_normalized_by_vcpu_count(self):
        self.assertAlmostEqual(
            normalized_cpu_percent(0, 2_000_000_000, 5.0, 2), 20.0
        )
        self.assertIsNone(normalized_cpu_percent(None, 2, 5.0, 2))

    def test_workload_request_enforces_global_memory_cap(self):
        with self.assertRaisesRegex(ValueError, str(MAX_TOTAL_MEMORY_MIB)):
            validate_workload_request(
                ["testvm1", "testvm2", "testvm3"],
                "memory_burst",
                10,
                512,
                10_000,
                {name: 4_000 for name in ("testvm1", "testvm2", "testvm3")},
            )

    def test_workload_request_enforces_host_and_guest_reserves(self):
        with self.assertRaisesRegex(ValueError, "Host MemAvailable"):
            validate_workload_request(
                ["testvm1"], "memory_burst", 10, 256, 2_100,
                {"testvm1": 4_000},
            )
        with self.assertRaisesRegex(ValueError, "guest MemAvailable"):
            validate_workload_request(
                ["testvm1"], "memory_burst", 10, 256, 10_000,
                {"testvm1": 1_100},
            )

    def test_campaign_schedule_balances_each_vm_workload_pair(self):
        counts = Counter()
        initial_vms, initial_profile = _next_vm_group(counts)
        self.assertEqual(initial_vms, ["testvm1", "testvm2", "testvm3"])
        self.assertEqual(initial_profile, WORKLOAD_PROFILES[0])
        for vm_name in initial_vms:
            counts[(vm_name, initial_profile)] += 24
        next_vms, next_profile = _next_vm_group(counts)
        self.assertNotEqual((next_vms, next_profile), (initial_vms, initial_profile))
        self.assertEqual(counts[(next_vms[0], next_profile)], 0)

    def test_campaign_groups_only_equally_collected_vm_profile_pairs(self):
        counts = Counter({
            (vm_name, profile): 100
            for vm_name in ("testvm1", "testvm2", "testvm3")
            for profile in WORKLOAD_PROFILES
        })
        counts[("testvm1", "idle")] = 10
        counts[("testvm2", "idle")] = 10
        counts[("testvm3", "idle")] = 20
        selected_vms, profile = _next_vm_group(counts)
        self.assertEqual(profile, "idle")
        self.assertEqual(selected_vms, ["testvm1", "testvm2"])

    def test_campaign_varies_memory_and_duration_between_repetitions(self):
        first = _workload_variant("memory_ramp", 0, 60, 256)
        second = _workload_variant("memory_ramp", 1, 60, 256)
        third = _workload_variant("memory_ramp", 2, 60, 256)
        self.assertEqual(first, (45.0, 128))
        self.assertEqual(second, (60, 192))
        self.assertEqual(third, (90.0, 256))


if __name__ == "__main__":
    unittest.main()
