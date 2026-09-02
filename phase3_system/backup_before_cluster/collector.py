"""
Polls libvirt for live memory statistics from a running guest VM.

Requires:
  - qemu-guest-agent installed and running inside the guest (for full stats)
  - the VM's memballoon device set to model='virtio'
"""

import collections
import time
import libvirt

import config


class MemoryStatsCollector:
    """
    Wraps a libvirt domain and keeps a rolling window of memory-stat samples.
    Each sample is a dict of the raw values libvirt's memoryStats() returns,
    plus a wall-clock timestamp.
    """

    def __init__(self, domain_name, uri=config.LIBVIRT_URI,
                 window_size=config.HISTORY_WINDOW_SIZE):
        self.conn = libvirt.open(uri)
        if self.conn is None:
            raise RuntimeError(f"Failed to open libvirt connection to {uri}")

        self.dom = self.conn.lookupByName(domain_name)
        if self.dom is None:
            raise RuntimeError(f"Domain '{domain_name}' not found")

        self.history = collections.deque(maxlen=window_size)

    def poll(self):
        """
        Pull one sample of memory stats and append to history.
        Returns the sample dict.

        Common keys returned by libvirt (availability depends on guest agent):
          actual       - current balloon target (KiB)
          swap_in      - KiB swapped in since boot
          swap_out     - KiB swapped out since boot
          major_fault  - major page faults
          minor_fault  - minor page faults
          unused       - KiB unused, as reported by guest OS
          available    - KiB available to the guest OS
          usable       - KiB the guest OS considers usable without swapping
          rss          - resident set size of the QEMU process (KiB), host-side
        """
        raw = self.dom.memoryStats()
        raw["_timestamp"] = time.time()

        # info() gives [state, maxMem, memory, nrVirtCpu, cpuTime]
        # 'memory' here is the current balloon target — useful cross-check against 'actual'
        info = self.dom.info()
        raw["_current_memory_kib"] = info[2]
        raw["_max_memory_kib"] = info[1]

        self.history.append(raw)
        return raw

    def get_history(self):
        return list(self.history)

    def is_ready(self):
        """History has enough samples to build a feature vector."""
        return len(self.history) == self.history.maxlen

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    # Quick manual test: python3 collector.py <domain_name>
    import sys
    import json

    domain = sys.argv[1] if len(sys.argv) > 1 else "testvm"
    c = MemoryStatsCollector(domain)
    for _ in range(5):
        sample = c.poll()
        print(json.dumps(sample, indent=2))
        time.sleep(config.POLL_INTERVAL_SECONDS)
    c.close()
