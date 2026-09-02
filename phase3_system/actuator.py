"""
Applies memory resize decisions to a live libvirt domain.

Prediction -> Decision -> Safe memory actuation
"""

import time
import libvirt

import config
from decision_engine import ACTION_INCREASE, ACTION_DECREASE, ACTION_NONE


class Actuator:
    def __init__(self, dom, dry_run=False):
        self.dom = dom
        self.dry_run = dry_run

    def apply(self, action, delta_kib, current_memory_kib):
        """
        Apply a memory action.

        Parameters:
            action: increase / decrease / none
            delta_kib: memory change in KiB
            current_memory_kib: current VM memory target in KiB

        Returns:
            (success, new_memory_kib, message)
        """

        # ----------------------------------------
        # NO ACTION
        # ----------------------------------------
        if action == ACTION_NONE or delta_kib == 0:
            return True, current_memory_kib, "no-op"

        # ----------------------------------------
        # DOMAIN SAFETY CHECK
        # ----------------------------------------
        try:
            if not self.dom.isActive():
                return (
                    False,
                    current_memory_kib,
                    "domain is not running — cannot change memory"
                )
        except libvirt.libvirtError as e:
            return (
                False,
                current_memory_kib,
                f"failed to check domain state: {e}"
            )

        # ----------------------------------------
        # CALCULATE NEW TARGET
        # ----------------------------------------
        requested_target = current_memory_kib + delta_kib

        # Get actual VM maximum memory from libvirt
        try:
            domain_info = self.dom.info()
            vm_max_memory_kib = domain_info[1]
        except libvirt.libvirtError:
            vm_max_memory_kib = config.MAX_MEMORY_KIB

        # Use the stricter ceiling
        safe_max_memory = min(
            config.MAX_MEMORY_KIB,
            vm_max_memory_kib
        )

        # Apply safety limits
        new_target = max(
            config.MIN_MEMORY_KIB,
            min(safe_max_memory, requested_target)
        )

        # ----------------------------------------
        # CHECK WHETHER TARGET ACTUALLY CHANGED
        # ----------------------------------------
        if new_target == current_memory_kib:
            return (
                True,
                current_memory_kib,
                "target unchanged after safety guardrails"
            )

        # ----------------------------------------
        # DRY RUN
        # ----------------------------------------
        if self.dry_run:
            return (
                True,
                new_target,
                f"[DRY RUN] would set memory "
                f"{current_memory_kib} KiB -> {new_target} KiB"
            )

        # ----------------------------------------
        # REAL MEMORY CHANGE
        # ----------------------------------------
        try:
            self.dom.setMemory(new_target)

            # Give libvirt a moment to process the request
            time.sleep(0.5)

            # Read back current balloon target
            updated_info = self.dom.info()
            actual_memory = updated_info[2]

            return (
                True,
                actual_memory,
                f"memory change requested: "
                f"{current_memory_kib} KiB -> {new_target} KiB "
                f"(reported current: {actual_memory} KiB)"
            )

        except libvirt.libvirtError as e:
            return (
                False,
                current_memory_kib,
                f"libvirt memory change failed: {e}"
            )
