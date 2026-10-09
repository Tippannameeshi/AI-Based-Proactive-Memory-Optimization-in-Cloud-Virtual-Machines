# Live Telemetry and Workload Experiments

## Architecture

This isolated path targets the running `testvm1`, `testvm2`, and `testvm3`
libvirt guests:

`run_live_experiment` → read-only libvirt CPU/balloon/interface counters plus
QEMU Guest Agent guest measurements → independent per-VM CSV rows → validator
and plots.

For non-idle profiles, the runner launches a bounded guest process through
QEMU Guest Agent, collects during the workload, sends SIGTERM and verifies
process exit, then collects the recovery phase. It does not load the existing
model, alter a VM's memory target, or start/stop a VM. The existing Phase 3
controller, actuator, decision engine, and model artifacts are not used.

## Requirements and safety

- Run from the repository root with Python that can import `libvirt` and access
  `qemu:///system` (or pass another URI).
- QEMU Guest Agent must be running in each selected guest. The shared helper
  invokes `virsh qemu-agent-command` without a shell, tracks the PID returned
  by `guest-exec`, polls `guest-exec-status`, and decodes captured base64 output.
  Guest telemetry is collected with `/bin/cat /proc/meminfo`; workload profiles
  use `/usr/bin/python3` through the agent. No shell is invoked in the guest.
- Workload allocations are limited to 512 MiB per guest and 768 MiB aggregate.
  The runner also preserves at least 2 GiB host `MemAvailable` and 1 GiB guest
  `MemAvailable` per selected guest. These are guardrails, not a guarantee against
  unrelated host/guest workloads consuming memory concurrently. Headroom is
  checked before baseline collection, immediately before the workload phase,
  and before each direct single-VM workload launch.
- `memory_burst`, `memory_ramp`, and `mixed` allocate resident memory in touched
  16 MiB chunks. CPU work is one guest process and is not pinned to a host core.
  Every launched process is sent SIGTERM after the workload phase, including on
  runner errors/interrupts. Start with one VM and a small allocation.
- Sampling is requested every 5 seconds by default. Collection takes time, so
  actual elapsed intervals are recorded and used for rates; do not assume exactly
  five seconds. First-sample rates and CPU utilization are blank because no prior
  counter exists.

## Run an experiment

```bash
python3 -m phase3_system.run_live_experiment \
  --vm testvm1 --profile memory_ramp --duration 120 --memory-mib 128 \
  --baseline-seconds 30 --recovery-seconds 60 --interval 5
```

Repeat `--vm` to select more guests. With no `--vm`, all three named VMs are
selected. Profiles are `idle`, `memory_burst`, `memory_ramp`, `cpu_burst`, and
`mixed`. `idle` performs no guest workload. The default output is
`phase3_system/data/live_telemetry.csv`; `--output` selects another path.
Collection appends only if an existing CSV has the exact same header and never
overwrites it.

## Balanced resumable campaign

To build a broad dataset, run the campaign manager. It schedules the VM/profile
pair with the fewest existing rows, preserves the existing CSV, rotates unique
32-character experiment IDs, and serializes writers with a file lock. Re-running
the same command resumes from recorded rows after a normal stop or interruption.
The final experiment is not split, so the count may slightly exceed the target.

```bash
python3 -m phase3_system.collect_live_dataset \
  --target-rows 20000 --duration 60 --baseline-seconds 30 \
  --recovery-seconds 30 --interval 5 --memory-mib 256
```

This starts live experiments and is not run automatically during setup or
validation. Progress-only reporting does not connect to libvirt or launch
workloads:

```bash
python3 -m phase3_system.collect_live_dataset --progress --target-rows 20000
```

The campaign balances all 15 VM/profile pairs. For the selected profile, it
groups the VMs tied at the least-collected pair count into one experiment.
Workloads run independently inside those selected guests at the same time;
the existing aggregate memory cap and host/guest headroom checks apply before
launch. Collection visits each VM sequentially within each sampling cycle,
using that VM's independent counters and actual elapsed time. It cycles through
45/60/90-second workload phases and 128/192/256 MiB per-VM memory caps (with
the default 60-second/256 MiB settings) for memory profiles. Safety headroom is
rechecked before every workload and may cause a workload to be refused.

When all three VMs are scheduled together and each yields one sample every
approximately five seconds, 20,000 rows require about 9.3 hours of observation
time, plus experiment setup and workload/recovery overhead. If current pair
coverage requires one or two VMs in some experiments, runtime falls between
that and the roughly 27.8-hour single-VM baseline. Stop/resume is expected.

## Telemetry definitions

- Every CSV observation includes a timezone-aware timestamp, experiment ID, VM
  ID, workload profile/phase/active marker, and
  `elapsed_seconds_since_previous_sample`.
- `guest_memory_used_mib` is exactly `(MemTotal - MemAvailable) / 1024`, using
  Linux guest `/proc/meminfo` values in KiB. It is not QEMU RSS, libvirt's
  allocation, or a balloon counter. Its denominator for
  `guest_memory_used_pct` is guest `MemTotal`.
- `balloon_current_mib` comes only from the balloon `actual` statistic and
  `balloon_maximum_mib` from libvirt's configured maximum. They are not the
  guest's Linux memory usage.
- Balloon measurements come from `virDomain.memoryStats()` and are blank when
  libvirt does not provide that statistic. They are not substituted for guest
  `/proc/meminfo`.
- `cpu_time_ns` is the cumulative domain CPU counter in nanoseconds. CPU utilization is
  the counter delta divided by actual elapsed time and vCPU count, times 100;
  100% means all configured vCPUs are busy.
- Interface byte counters are summed over target devices discovered from domain
  XML. `network_rx_bytes_per_sec` and `network_tx_bytes_per_sec` use counter
  deltas divided by actual elapsed monotonic time. Missing
  or failed measurements are blank and described in `collection_errors`, never
  fabricated as zero.

The CSV columns, in order, are:

`timestamp`, `experiment_id`, `vm_id`, `workload_profile`, `phase`,
`workload_active`, `elapsed_seconds_since_previous_sample`,
`guest_memory_total_mib`,
`guest_memory_available_mib`, `guest_memory_used_mib`,
`guest_memory_used_pct`, `guest_swap_total_mib`, `guest_swap_free_mib`,
`guest_swap_used_mib`, `balloon_current_mib`, `balloon_maximum_mib`,
`balloon_unused_mib`, `balloon_available_mib`, `balloon_usable_mib`,
`balloon_rss_mib`, `balloon_swap_in_mib`, `balloon_swap_out_mib`, `vcpu_count`,
`cpu_time_ns`, `cpu_utilization_pct`, `network_rx_bytes`, `network_tx_bytes`,
`network_rx_bytes_per_sec`, `network_tx_bytes_per_sec`, `collection_errors`.

## Workload profiles

- `idle`: no guest process; collects an idle comparison phase.
- `memory_burst`: allocates and touches the selected memory limit immediately.
- `memory_ramp`: ramps memory up, holds it, then releases it gradually.
- `cpu_burst`: one guest-side CPU-bound process; no memory allocation workload.
- `mixed`: combines the ramped memory and CPU workloads.

All non-idle worker processes end after the configured duration; the runner also
checks and terminates any process still running before entering recovery.

## Validate and plot

```bash
python3 -m phase3_system.validate_live_telemetry \
  phase3_system/data/live_telemetry.csv --target-rows 20000
python3 -m phase3_system.plot_live_telemetry \
  --input phase3_system/data/live_telemetry.csv
```

The validator reports row/VM/workload/phase/experiment counts, missing values,
collection errors, and within-experiment intervals. It checks exact schema,
per-VM coverage, timestamps, ranges, memory/swap arithmetic, and rates, and
returns failure if data errors are found. It does not create rolling or lag
features. Pass repeated `--vm` arguments to match the VMs present in a
single-VM CSV; without them, the validator requires samples from all three
configured VMs. The plotter creates guest-used-memory, guest-available-memory,
CPU, and combined network RX/TX plots under `phase3_system/data/plots/` by default.

## Not a model/controller integration

This dataset intentionally contains only measurements reproducible from the
host/libvirt/guest-agent environment. It does not recreate the original
33-feature LSTM training contract, infer a training cadence, train a model,
resize memory, or decide when a VM should be resized. The requested interval is
not guaranteed to be an exact cadence; actual elapsed intervals are recorded.
A future model must be trained and evaluated against this same live feature
definition and documented sampling cadence before any controller integration.
