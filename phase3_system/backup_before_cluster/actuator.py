"""
Applies a memory resize decision to a live domain via libvirt's setMemory().
This is the actual "actuation" step — where prediction becomes real balloon movement.
"""

import libvirt

import config
from decision_engine import ACTION_INCREASE, ACTION_DECREASE, ACTION_NONE


class Actuator:
    def __init__(self, dom, dry_run=False):
        self.dom = dom
        self.dry_run = dry_run

    def apply(self, action, delta_kib, current_memory_kib):
        """
        Returns: (success: bool, new_target_kib: int, message: str)
        """
        if action == ACTION_NONE or delta_kib == 0:
            return True, current_memory_kib, "no-op"

        new_target = current_memory_kib + delta_kib
        new_target = max(config.MIN_MEMORY_KIB, min(config.MAX_MEMORY_KIB, new_target))

        if self.dry_run:
            return True, new_target, f"[DRY RUN] would set memory to {new_target} KiB"

        try:
            # setMemory expects KiB and affects the live balloon target
            self.dom.setMemory(new_target)
            return True, new_target, f"memory set to {new_target} KiB"
        except libvirt.libvirtError as e:
            return False, current_memory_kib, f"libvirt error: {e}"
