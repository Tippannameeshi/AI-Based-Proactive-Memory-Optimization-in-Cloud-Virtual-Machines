#!/usr/bin/env python3

import argparse
import time
import sqlite3
import math
from datetime import datetime

import libvirt


# ============================================================
# CONFIGURATION
# ============================================================

DOMAINS = ["testvm1", "testvm2", "testvm3"]

POLL_INTERVAL = 5

# VM becomes receiver at or above this usage
RECEIVER_THRESHOLD = 75.0

# VM can donate at or below this usage
DONOR_THRESHOLD = 70.0

# Target receiver usage after allocation
TARGET_RECEIVER_USAGE = 70.0

# Safety limit ONLY for donating memory
SAFE_POST_DONATION_USAGE = 85.0

MEMORY_STEP_MB = 256
MIN_TRANSFER_MB = 32

MIN_MEMORY_MB = 1024
MAX_MEMORY_MB = 4096

ROLE_COOLDOWN_CYCLES = 2

DATABASE_FILE = "phase3_memory.db"


# ============================================================
# CENTRALIZED MULTI-VM MEMORY ORCHESTRATOR
# ============================================================

class CentralMemoryOrchestrator:

    def __init__(self, dry_run=False):

        self.dry_run = dry_run

        self.conn = None
        self.db_conn = None

        self.last_transfer_cycle = {}
        self.previous_roles = {}


    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    def init_database(self):

        self.db_conn = sqlite3.connect(DATABASE_FILE)

        cursor = self.db_conn.cursor()

        # VM METRICS

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vm_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle INTEGER,
                vm_name TEXT NOT NULL,

                allocated_memory_mb INTEGER,
                used_memory_mb INTEGER,
                available_memory_mb INTEGER,
                unused_memory_mb INTEGER,

                usage_percent REAL,

                rss_mb INTEGER,
                metric TEXT,

                role TEXT
            )
        """)

        # MEMORY TRANSFERS

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_transfers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle INTEGER,

                vm_name TEXT NOT NULL,

                transfer_type TEXT NOT NULL,

                old_memory_mb INTEGER,
                new_memory_mb INTEGER,
                delta_mb INTEGER,

                success INTEGER,
                dry_run INTEGER
            )
        """)

        # ORCHESTRATION CYCLES

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orchestration_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,
                cycle INTEGER UNIQUE,

                total_vms INTEGER,

                donor_count INTEGER,
                receiver_count INTEGER,
                cooldown_count INTEGER,

                total_allocated_memory_mb INTEGER,

                total_memory_collected_mb INTEGER,
                total_memory_distributed_mb INTEGER,

                actions_taken INTEGER,

                status TEXT
            )
        """)

        # ROLE HISTORY

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vm_role_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,
                cycle INTEGER,

                vm_name TEXT NOT NULL,

                previous_role TEXT,
                current_role TEXT,

                usage_percent REAL,

                reason TEXT
            )
        """)

        # SYSTEM EVENTS

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,
                cycle INTEGER,

                event_type TEXT NOT NULL,

                vm_name TEXT,

                message TEXT,

                severity TEXT
            )
        """)

        # CONFIGURATION

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threshold_configuration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,

                receiver_threshold REAL,
                donor_threshold REAL,

                receiver_target_usage REAL,

                safe_post_donation_usage REAL,

                memory_step_mb INTEGER,

                min_transfer_mb INTEGER,

                min_memory_mb INTEGER,
                max_memory_mb INTEGER,

                role_cooldown_cycles INTEGER,

                poll_interval_seconds INTEGER,

                dry_run INTEGER
            )
        """)

        # ====================================================
        # MEMORY LOANS
        #
        # Tracks:
        # donor -> receiver
        # amount borrowed
        # receiver usage when loan was created
        #
        # Memory is returned ONLY when actual workload usage
        # decreases after borrowing.
        # ====================================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_loans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_timestamp TEXT NOT NULL,
                returned_timestamp TEXT,

                created_cycle INTEGER,
                returned_cycle INTEGER,

                donor_vm TEXT NOT NULL,
                receiver_vm TEXT NOT NULL,

                original_amount_mb INTEGER NOT NULL,
                remaining_amount_mb INTEGER NOT NULL,

                receiver_used_mb_at_loan INTEGER NOT NULL,

                status TEXT NOT NULL
            )
        """)

        self.db_conn.commit()

        self.log_configuration()

        print(
            f"[database] SQLite database initialized: "
            f"{DATABASE_FILE}"
        )

        print(
            "[database] Tables ready: "
            "vm_metrics, memory_transfers, "
            "orchestration_cycles, vm_role_history, "
            "system_events, threshold_configuration, "
            "memory_loans"
        )


    # ========================================================
    # CLOSE DATABASE
    # ========================================================

    def close_database(self):

        if self.db_conn:

            self.db_conn.close()

            print(
                "[database] SQLite connection closed."
            )


    # ========================================================
    # LOG CONFIGURATION
    # ========================================================

    def log_configuration(self):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            INSERT INTO threshold_configuration (
                timestamp,

                receiver_threshold,
                donor_threshold,

                receiver_target_usage,

                safe_post_donation_usage,

                memory_step_mb,

                min_transfer_mb,

                min_memory_mb,
                max_memory_mb,

                role_cooldown_cycles,

                poll_interval_seconds,

                dry_run
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            RECEIVER_THRESHOLD,
            DONOR_THRESHOLD,

            TARGET_RECEIVER_USAGE,

            SAFE_POST_DONATION_USAGE,

            MEMORY_STEP_MB,

            MIN_TRANSFER_MB,

            MIN_MEMORY_MB,
            MAX_MEMORY_MB,

            ROLE_COOLDOWN_CYCLES,

            POLL_INTERVAL,

            1 if self.dry_run else 0
        ))

        self.db_conn.commit()


    # ========================================================
    # LOG VM METRICS
    # ========================================================

    def log_vm_metrics(self, cycle, vm, role):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            INSERT INTO vm_metrics (
                timestamp,
                cycle,
                vm_name,

                allocated_memory_mb,
                used_memory_mb,
                available_memory_mb,
                unused_memory_mb,

                usage_percent,

                rss_mb,
                metric,

                role
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            cycle,

            vm["name"],

            vm["memory_mb"],

            int(vm["used_kib"] / 1024),

            int(vm["available_kib"] / 1024),

            int(vm["unused_kib"] / 1024),

            vm["usage_percent"],

            vm["rss_mb"],

            vm["metric"],

            role
        ))

        self.db_conn.commit()


    # ========================================================
    # LOG TRANSFER
    # ========================================================

    def log_transfer(
        self,
        cycle,
        vm_name,
        transfer_type,
        old_memory_mb,
        new_memory_mb,
        delta_mb,
        success
    ):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            INSERT INTO memory_transfers (
                timestamp,
                cycle,

                vm_name,

                transfer_type,

                old_memory_mb,
                new_memory_mb,
                delta_mb,

                success,
                dry_run
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            cycle,

            vm_name,

            transfer_type,

            old_memory_mb,
            new_memory_mb,
            delta_mb,

            1 if success else 0,

            1 if self.dry_run else 0
        ))

        self.db_conn.commit()


    # ========================================================
    # LOG EVENT
    # ========================================================

    def log_event(
        self,
        cycle,
        event_type,
        message,
        vm_name=None,
        severity="INFO"
    ):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            INSERT INTO system_events (
                timestamp,
                cycle,

                event_type,

                vm_name,

                message,

                severity
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            cycle,

            event_type,

            vm_name,

            message,

            severity
        ))

        self.db_conn.commit()


    # ========================================================
    # LOG CYCLE SUMMARY
    # ========================================================

    def log_cycle_summary(
        self,
        cycle,
        vm_stats,
        donors,
        receivers,
        cooldown_vms,
        memory_collected,
        memory_distributed,
        actions_taken,
        status
    ):

        cursor = self.db_conn.cursor()

        total_allocated_memory = sum(
            vm["memory_mb"]
            for vm in vm_stats
        )

        cursor.execute("""
            INSERT OR REPLACE INTO orchestration_cycles (
                timestamp,
                cycle,

                total_vms,

                donor_count,
                receiver_count,
                cooldown_count,

                total_allocated_memory_mb,

                total_memory_collected_mb,
                total_memory_distributed_mb,

                actions_taken,

                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            cycle,

            len(vm_stats),

            len(donors),
            len(receivers),
            len(cooldown_vms),

            total_allocated_memory,

            memory_collected,
            memory_distributed,

            actions_taken,

            status
        ))

        self.db_conn.commit()


    # ========================================================
    # ROLE HISTORY
    # ========================================================

    def log_role_history(
        self,
        cycle,
        vm,
        current_role
    ):

        vm_name = vm["name"]

        previous_role = self.previous_roles.get(
            vm_name,
            "UNKNOWN"
        )

        if previous_role != current_role:

            cursor = self.db_conn.cursor()

            cursor.execute("""
                INSERT INTO vm_role_history (
                    timestamp,
                    cycle,

                    vm_name,

                    previous_role,
                    current_role,

                    usage_percent,

                    reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                cycle,

                vm_name,

                previous_role,
                current_role,

                vm["usage_percent"],

                "VM role classification updated"
            ))

            self.db_conn.commit()

            self.previous_roles[
                vm_name
            ] = current_role


    # ========================================================
    # MEMORY LOAN FUNCTIONS
    # ========================================================

    def create_memory_loan(
        self,
        cycle,
        donor_vm,
        receiver_vm,
        amount_mb,
        receiver_used_mb
    ):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            INSERT INTO memory_loans (
                created_timestamp,
                created_cycle,

                donor_vm,
                receiver_vm,

                original_amount_mb,
                remaining_amount_mb,

                receiver_used_mb_at_loan,

                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            cycle,

            donor_vm,
            receiver_vm,

            amount_mb,
            amount_mb,

            int(receiver_used_mb),

            "ACTIVE"
        ))

        self.db_conn.commit()

        print(
            f"[loan] Recorded: "
            f"{donor_vm} -> {receiver_vm} "
            f"({amount_mb} MB)"
        )


    def get_active_loans(self, receiver_vm):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            SELECT
                id,
                donor_vm,
                receiver_vm,
                remaining_amount_mb,
                receiver_used_mb_at_loan

            FROM memory_loans

            WHERE receiver_vm = ?
            AND status = 'ACTIVE'
            AND remaining_amount_mb > 0

            ORDER BY id ASC
        """, (
            receiver_vm,
        ))

        return cursor.fetchall()


    def return_loan_amount(
        self,
        loan_id,
        amount_mb,
        cycle
    ):

        cursor = self.db_conn.cursor()

        cursor.execute("""
            SELECT remaining_amount_mb
            FROM memory_loans
            WHERE id = ?
        """, (
            loan_id,
        ))

        row = cursor.fetchone()

        if not row:
            return

        remaining = row[0]

        new_remaining = max(
            0,
            remaining - amount_mb
        )

        status = (
            "RETURNED"
            if new_remaining == 0
            else "PARTIALLY_RETURNED"
        )

        returned_timestamp = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if new_remaining == 0
            else None
        )

        cursor.execute("""
            UPDATE memory_loans
            SET
                remaining_amount_mb = ?,
                status = ?,
                returned_cycle = ?,
                returned_timestamp = ?
            WHERE id = ?
        """, (
            new_remaining,
            status,
            cycle,
            returned_timestamp,
            loan_id
        ))

        self.db_conn.commit()


    # ========================================================
    # LIBVIRT
    # ========================================================

    def connect(self):

        self.conn = libvirt.open(
            "qemu:///system"
        )

        if self.conn is None:

            raise RuntimeError(
                "Failed to connect to libvirt"
            )

        print(
            "[central] Connected to libvirt"
        )


    def close(self):

        if self.conn:

            self.conn.close()

            print(
                "[central] Libvirt connection closed."
            )


    # ========================================================
    # GET VM STATISTICS
    # ========================================================

    def get_vm_stats(self, domain_name):

        domain = self.conn.lookupByName(
            domain_name
        )

        info = domain.info()

        current_memory_kib = info[2]

        stats = domain.memoryStats()

        available_kib = stats.get(
            "available",
            0
        )

        unused_kib = stats.get(
            "unused",
            0
        )

        rss_kib = stats.get(
            "rss",
            0
        )

        # Prefer RSS

        if (
            rss_kib > 0
            and current_memory_kib > 0
        ):

            used_kib = min(
                rss_kib,
                current_memory_kib
            )

            usage_percent = (
                used_kib /
                current_memory_kib
            ) * 100

            metric = "RSS"

        elif available_kib > 0:

            used_kib = max(
                0,
                available_kib - unused_kib
            )

            usage_percent = (
                used_kib /
                available_kib
            ) * 100

            metric = "AVAILABLE-UNUSED"

        else:

            used_kib = 0
            usage_percent = 50.0
            metric = "FALLBACK"

        return {
            "name": domain_name,
            "domain": domain,

            "memory_kib": current_memory_kib,

            "memory_mb": int(
                current_memory_kib / 1024
            ),

            "available_kib": available_kib,

            "unused_kib": unused_kib,

            "used_kib": used_kib,

            "usage_percent": usage_percent,

            "rss_kib": rss_kib,

            "rss_mb": int(
                rss_kib / 1024
            ),

            "metric": metric
        }


    # ========================================================
    # COOLDOWN
    # ========================================================

    def is_in_cooldown(
        self,
        vm_name,
        cycle
    ):

        if vm_name not in self.last_transfer_cycle:
            return False

        last_cycle = (
            self.last_transfer_cycle[vm_name]
        )

        return (
            cycle - last_cycle
            <= ROLE_COOLDOWN_CYCLES
        )


    # ========================================================
    # SAFE DONATION
    # ========================================================

    def calculate_safe_donation(self, vm):

        current_memory_mb = vm["memory_mb"]

        used_memory_mb = (
            vm["used_kib"] / 1024
        )

        absolute_available = max(
            0,
            current_memory_mb - MIN_MEMORY_MB
        )

        if absolute_available <= 0:
            return 0

        if used_memory_mb <= 0:
            return int(absolute_available)

        minimum_safe_memory = math.ceil(
            used_memory_mb /
            (
                SAFE_POST_DONATION_USAGE / 100.0
            )
        )

        minimum_required_memory = max(
            MIN_MEMORY_MB,
            minimum_safe_memory
        )

        safe_donation = (
            current_memory_mb -
            minimum_required_memory
        )

        return max(
            0,
            int(min(
                safe_donation,
                absolute_available
            ))
        )


    # ========================================================
    # RECEIVER NEED
    # ========================================================

    def calculate_receiver_need(self, vm):

        current_memory_mb = vm["memory_mb"]

        used_memory_mb = (
            vm["used_kib"] / 1024
        )

        if used_memory_mb <= 0:
            return 0

        target_memory = math.ceil(
            used_memory_mb /
            (
                TARGET_RECEIVER_USAGE / 100.0
            )
        )

        target_memory = min(
            MAX_MEMORY_MB,
            target_memory
        )

        return max(
            0,
            int(
                target_memory -
                current_memory_mb
            )
        )


    # ========================================================
    # CLASSIFY VMS
    # ========================================================

    def classify_vms(
        self,
        vm_stats,
        cycle
    ):

        donors = []
        receivers = []
        cooldown_vms = []

        for vm in vm_stats:

            usage = vm["usage_percent"]
            vm_name = vm["name"]

            if self.is_in_cooldown(
                vm_name,
                cycle
            ):

                cooldown_vms.append(vm)
                continue

            if (
                usage >= RECEIVER_THRESHOLD
                and vm["memory_mb"] < MAX_MEMORY_MB
            ):

                receivers.append(vm)
                continue

            if (
                usage <= DONOR_THRESHOLD
                and vm["memory_mb"] > MIN_MEMORY_MB
            ):

                safe_donation = (
                    self.calculate_safe_donation(vm)
                )

                if safe_donation >= MIN_TRANSFER_MB:
                    donors.append(vm)

        return (
            donors,
            receivers,
            cooldown_vms
        )


    # ========================================================
    # REDUCE DONOR MEMORY
    # ========================================================

    def reduce_memory(
        self,
        vm,
        amount_mb,
        cycle
    ):

        old_memory_mb = vm["memory_mb"]

        safe_donation = (
            self.calculate_safe_donation(vm)
        )

        actual_delta_mb = min(
            amount_mb,
            safe_donation,
            MEMORY_STEP_MB
        )

        if actual_delta_mb < MIN_TRANSFER_MB:
            return 0

        new_memory_mb = (
            old_memory_mb -
            actual_delta_mb
        )

        if new_memory_mb < MIN_MEMORY_MB:

            new_memory_mb = MIN_MEMORY_MB

            actual_delta_mb = (
                old_memory_mb -
                new_memory_mb
            )

        if actual_delta_mb < MIN_TRANSFER_MB:
            return 0

        used_memory_mb = (
            vm["used_kib"] / 1024
        )

        predicted_usage = (
            used_memory_mb /
            new_memory_mb
        ) * 100

        if predicted_usage > SAFE_POST_DONATION_USAGE:

            print(
                f"[planner] {vm['name']} "
                f"donation rejected for safety"
            )

            return 0

        print(
            f"[transfer] DONOR {vm['name']}: "
            f"{old_memory_mb} MB -> "
            f"{new_memory_mb} MB "
            f"(-{actual_delta_mb} MB)"
        )

        success = True

        if not self.dry_run:

            try:

                vm["domain"].setMemory(
                    new_memory_mb * 1024
                )

                print(
                    f"[actuator] Successfully reduced "
                    f"{vm['name']} memory"
                )

            except libvirt.libvirtError as e:

                print(
                    f"[ERROR] Failed reducing "
                    f"{vm['name']}: {e}"
                )

                success = False

        else:

            print(
                f"[DRY RUN] Would reduce "
                f"{vm['name']} by "
                f"{actual_delta_mb} MB"
            )

        self.log_transfer(
            cycle,
            vm["name"],
            "DONATE",
            old_memory_mb,
            new_memory_mb,
            -actual_delta_mb,
            success
        )

        if not success:
            return 0

        self.last_transfer_cycle[
            vm["name"]
        ] = cycle

        return actual_delta_mb


    # ========================================================
    # INCREASE RECEIVER MEMORY
    # ========================================================

    def increase_memory(
        self,
        vm,
        amount_mb,
        cycle
    ):

        old_memory_mb = vm["memory_mb"]

        receiver_capacity = (
            MAX_MEMORY_MB -
            old_memory_mb
        )

        actual_delta_mb = min(
            amount_mb,
            receiver_capacity
        )

        if actual_delta_mb < MIN_TRANSFER_MB:
            return 0

        new_memory_mb = (
            old_memory_mb +
            actual_delta_mb
        )

        print(
            f"[transfer] RECEIVER {vm['name']}: "
            f"{old_memory_mb} MB -> "
            f"{new_memory_mb} MB "
            f"(+{actual_delta_mb} MB)"
        )

        success = True

        if not self.dry_run:

            try:

                vm["domain"].setMemory(
                    new_memory_mb * 1024
                )

                print(
                    f"[actuator] Successfully increased "
                    f"{vm['name']} memory"
                )

            except libvirt.libvirtError as e:

                print(
                    f"[ERROR] Failed increasing "
                    f"{vm['name']}: {e}"
                )

                success = False

        else:

            print(
                f"[DRY RUN] Would increase "
                f"{vm['name']} by "
                f"{actual_delta_mb} MB"
            )

        self.log_transfer(
            cycle,
            vm["name"],
            "RECEIVE",
            old_memory_mb,
            new_memory_mb,
            actual_delta_mb,
            success
        )

        if not success:
            return 0

        self.last_transfer_cycle[
            vm["name"]
        ] = cycle

        return actual_delta_mb


    # ========================================================
    # WORKLOAD-BASED MEMORY RECLAMATION
    #
    # IMPORTANT:
    # NO RECLAMATION THRESHOLD
    # NO 85% RECLAMATION RULE
    #
    # Memory returns ONLY if actual used memory decreases
    # compared with usage when the loan was created.
    # ========================================================

    def reclaim_memory(
        self,
        vm_stats,
        cycle
    ):

        print(
            "\n[reclamation] Checking active memory loans..."
        )

        stats_map = {
            vm["name"]: vm
            for vm in vm_stats
        }

        total_reclaimed = 0
        actions = 0

        for receiver in vm_stats:

            receiver_name = receiver["name"]

            active_loans = self.get_active_loans(
                receiver_name
            )

            if not active_loans:
                continue

            current_used_mb = (
                receiver["used_kib"] / 1024
            )

            for loan in active_loans:

                loan_id = loan[0]
                donor_name = loan[1]
                remaining_loan = loan[3]
                used_at_loan = loan[4]

                if donor_name not in stats_map:
                    continue

                donor = stats_map[donor_name]

                # ============================================
                # CORE RECLAMATION LOGIC
                #
                # Only reclaim if REAL workload usage drops.
                # ============================================

                freed_memory = (
                    used_at_loan -
                    current_used_mb
                )

                if freed_memory <= 0:

                    print(
                        f"[reclamation] "
                        f"{receiver_name}: "
                        f"workload still active"
                    )

                    continue

                return_amount = min(
                    remaining_loan,
                    int(freed_memory)
                )

                max_possible_return = max(
                    0,
                    receiver["memory_mb"] -
                    MIN_MEMORY_MB
                )

                return_amount = min(
                    return_amount,
                    max_possible_return
                )

                if return_amount < MIN_TRANSFER_MB:
                    continue

                donor_capacity = (
                    MAX_MEMORY_MB -
                    donor["memory_mb"]
                )

                return_amount = min(
                    return_amount,
                    donor_capacity
                )

                if return_amount < MIN_TRANSFER_MB:
                    continue

                old_receiver_memory = (
                    receiver["memory_mb"]
                )

                old_donor_memory = (
                    donor["memory_mb"]
                )

                new_receiver_memory = (
                    old_receiver_memory -
                    return_amount
                )

                new_donor_memory = (
                    old_donor_memory +
                    return_amount
                )

                print(
                    f"[reclamation] Actual workload "
                    f"decreased on {receiver_name}"
                )

                print(
                    f"[reclamation] Returning "
                    f"{return_amount} MB: "
                    f"{receiver_name} -> "
                    f"{donor_name}"
                )

                success = True

                if not self.dry_run:

                    try:

                        receiver["domain"].setMemory(
                            new_receiver_memory * 1024
                        )

                        donor["domain"].setMemory(
                            new_donor_memory * 1024
                        )

                        print(
                            f"[reclamation] Successfully "
                            f"returned {return_amount} MB"
                        )

                    except libvirt.libvirtError as e:

                        print(
                            f"[ERROR] Reclamation failed: "
                            f"{e}"
                        )

                        success = False

                else:

                    print(
                        f"[DRY RUN] Would return "
                        f"{return_amount} MB"
                    )

                self.log_transfer(
                    cycle,
                    receiver_name,
                    "RECLAIM_OUT",
                    old_receiver_memory,
                    new_receiver_memory,
                    -return_amount,
                    success
                )

                self.log_transfer(
                    cycle,
                    donor_name,
                    "RECLAIM_RETURN",
                    old_donor_memory,
                    new_donor_memory,
                    return_amount,
                    success
                )

                if success:

                    self.return_loan_amount(
                        loan_id,
                        return_amount,
                        cycle
                    )

                    self.log_event(
                        cycle,
                        "MEMORY_RETURNED",
                        (
                            f"{return_amount} MB returned "
                            f"from {receiver_name} "
                            f"to {donor_name} after "
                            f"actual workload decrease"
                        ),
                        receiver_name
                    )

                    total_reclaimed += return_amount
                    actions += 2

                    receiver["memory_mb"] = (
                        new_receiver_memory
                    )

                    donor["memory_mb"] = (
                        new_donor_memory
                    )

                    self.last_transfer_cycle[
                        receiver_name
                    ] = cycle

                    self.last_transfer_cycle[
                        donor_name
                    ] = cycle

        if total_reclaimed == 0:

            print(
                "[reclamation] No workload-based "
                "memory return required"
            )

        else:

            print(
                f"[reclamation] Total memory returned: "
                f"{total_reclaimed} MB"
            )

        return total_reclaimed, actions


    # ========================================================
    # PLAN AND EXECUTE REALLOCATION
    # ========================================================

    def plan_and_execute(
        self,
        donors,
        receivers,
        cycle
    ):

        result = {
            "collected": 0,
            "distributed": 0,
            "actions": 0,
            "status": "STABLE"
        }

        if not receivers:

            print(
                "[planner] No receiver VMs require memory"
            )

            return result

        if not donors:

            print(
                "[planner] No safe donor VMs available"
            )

            result["status"] = "RESOURCE_SHORTAGE"

            return result

        result["status"] = "REALLOCATION"

        print(
            "\n[planner] Memory reallocation plan"
        )

        donors.sort(
            key=lambda vm: vm["usage_percent"]
        )

        receivers.sort(
            key=lambda vm: vm["usage_percent"],
            reverse=True
        )

        # ================================================
        # Collect memory
        # ================================================

        total_donated = 0
        donor_contributions = []

        print(
            "\n[planner] Collecting memory from donors..."
        )

        for donor in donors:

            safe_amount = (
                self.calculate_safe_donation(
                    donor
                )
            )

            donation_amount = min(
                MEMORY_STEP_MB,
                safe_amount
            )

            donated = self.reduce_memory(
                donor,
                donation_amount,
                cycle
            )

            if donated > 0:

                total_donated += donated

                donor_contributions.append({
                    "donor": donor["name"],
                    "amount": donated
                })

                result["actions"] += 1

        result["collected"] = total_donated

        print(
            f"\n[planner] Total memory collected: "
            f"{total_donated} MB"
        )

        if total_donated < MIN_TRANSFER_MB:

            result["status"] = "TRANSFER_FAILED"

            return result

        # ================================================
        # Distribute memory
        # ================================================

        print(
            "\n[planner] Distributing memory to receivers..."
        )

        remaining_memory = total_donated

        contribution_index = 0

        for receiver in receivers:

            if remaining_memory < MIN_TRANSFER_MB:
                break

            receiver_need = (
                self.calculate_receiver_need(
                    receiver
                )
            )

            if receiver_need <= 0:
                continue

            allocation = min(
                remaining_memory,
                receiver_need
            )

            if allocation < MIN_TRANSFER_MB:
                continue

            received = self.increase_memory(
                receiver,
                allocation,
                cycle
            )

            if received <= 0:
                continue

            result["distributed"] += received
            result["actions"] += 1

            remaining_memory -= received

            # ============================================
            # Record donor -> receiver loan relationship
            # ============================================

            amount_to_record = received

            while (
                amount_to_record > 0
                and contribution_index <
                len(donor_contributions)
            ):

                contribution = (
                    donor_contributions[
                        contribution_index
                    ]
                )

                available_from_donor = (
                    contribution["amount"]
                )

                loan_amount = min(
                    amount_to_record,
                    available_from_donor
                )

                self.create_memory_loan(
                    cycle,
                    contribution["donor"],
                    receiver["name"],
                    loan_amount,
                    receiver["used_kib"] / 1024
                )

                contribution["amount"] -= (
                    loan_amount
                )

                amount_to_record -= loan_amount

                if contribution["amount"] == 0:
                    contribution_index += 1

        print()

        print(
            "[central] Memory reallocation cycle complete"
        )

        return result


    # ========================================================
    # MAIN LOOP
    # ========================================================

    def run(self):

        self.connect()

        self.init_database()

        print()
        print("=" * 65)
        print(" CENTRALIZED MULTI-VM MEMORY ORCHESTRATOR")
        print("=" * 65)

        print(
            f"\nDomains: {', '.join(DOMAINS)}"
        )

        mode = (
            "DRY RUN"
            if self.dry_run
            else "LIVE MODE"
        )

        print(f"Mode: {mode}")
        print(f"Poll interval: {POLL_INTERVAL}s")

        print(
            f"Receiver threshold: >= "
            f"{RECEIVER_THRESHOLD}%"
        )

        print(
            f"Donor threshold: <= "
            f"{DONOR_THRESHOLD}%"
        )

        print(
            f"Receiver target usage: "
            f"{TARGET_RECEIVER_USAGE}%"
        )

        print(
            f"Safe post-transfer usage: <= "
            f"{SAFE_POST_DONATION_USAGE}%"
        )

        print()
        print("Memory reclamation: ENABLED")

        print(
            "Reclamation policy: "
            "Actual workload decrease"
        )

        print(
            "(No separate reclamation threshold)"
        )

        print()
        print(
            f"Database: {DATABASE_FILE}"
        )

        print(
            "\nPress Ctrl+C to stop."
        )

        cycle = 0

        try:

            while True:

                cycle += 1

                print()
                print("=" * 65)

                print(
                    f"[CENTRAL CYCLE {cycle}] "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                print("=" * 65)

                # ============================================
                # GET INITIAL VM STATS
                # ============================================

                vm_stats = []

                for domain_name in DOMAINS:

                    try:

                        stats = self.get_vm_stats(
                            domain_name
                        )

                        vm_stats.append(stats)

                    except libvirt.libvirtError as e:

                        print(
                            f"[ERROR] Could not read "
                            f"{domain_name}: {e}"
                        )

                # ============================================
                # RECLAMATION FIRST
                # ============================================

                self.reclaim_memory(
                    vm_stats,
                    cycle
                )

                # ============================================
                # REFRESH STATS AFTER RECLAMATION
                #
                # IMPORTANT because memory allocation may
                # have changed.
                # ============================================

                vm_stats = []

                for domain_name in DOMAINS:

                    try:

                        stats = self.get_vm_stats(
                            domain_name
                        )

                        vm_stats.append(stats)

                    except libvirt.libvirtError as e:

                        print(
                            f"[ERROR] Could not refresh "
                            f"{domain_name}: {e}"
                        )

                # ============================================
                # CLASSIFY VMS
                # ============================================

                (
                    donors,
                    receivers,
                    cooldown_vms
                ) = self.classify_vms(
                    vm_stats,
                    cycle
                )

                donor_names = {
                    vm["name"]
                    for vm in donors
                }

                receiver_names = {
                    vm["name"]
                    for vm in receivers
                }

                cooldown_names = {
                    vm["name"]
                    for vm in cooldown_vms
                }

                # ============================================
                # LOG METRICS
                # ============================================

                for vm in vm_stats:

                    if vm["name"] in receiver_names:
                        role = "RECEIVER"

                    elif vm["name"] in donor_names:
                        role = "DONOR"

                    elif vm["name"] in cooldown_names:
                        role = "COOLDOWN"

                    else:
                        role = "SAFE"

                    self.log_vm_metrics(
                        cycle,
                        vm,
                        role
                    )

                    self.log_role_history(
                        cycle,
                        vm,
                        role
                    )

                # ============================================
                # DISPLAY STATUS
                # ============================================

                print("\nVM STATUS")
                print("-" * 110)

                for vm in vm_stats:

                    print(
                        f"{vm['name']:<10} | "
                        f"Allocated: "
                        f"{vm['memory_mb']:>4} MB | "
                        f"Usage: "
                        f"{vm['usage_percent']:>6.2f}% | "
                        f"Used: "
                        f"{int(vm['used_kib']/1024):>4} MB | "
                        f"RSS: "
                        f"{vm['rss_mb']:>4} MB | "
                        f"Metric: "
                        f"{vm['metric']}"
                    )

                print()

                print(
                    "Donors:    " +
                    (
                        str([
                            vm["name"]
                            for vm in donors
                        ])
                        if donors else "None"
                    )
                )

                print(
                    "Receivers: " +
                    (
                        str([
                            vm["name"]
                            for vm in receivers
                        ])
                        if receivers else "None"
                    )
                )

                print(
                    "Cooldown:  " +
                    (
                        str([
                            vm["name"]
                            for vm in cooldown_vms
                        ])
                        if cooldown_vms else "None"
                    )
                )

                # ============================================
                # MEMORY REALLOCATION
                # ============================================

                result = self.plan_and_execute(
                    donors,
                    receivers,
                    cycle
                )

                # ============================================
                # LOG SUMMARY
                # ============================================

                self.log_cycle_summary(
                    cycle,
                    vm_stats,
                    donors,
                    receivers,
                    cooldown_vms,
                    result["collected"],
                    result["distributed"],
                    result["actions"],
                    result["status"]
                )

                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:

            print(
                "\n[central] Stopped safely by user."
            )

        finally:

            self.close_database()
            self.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Centralized Multi-VM "
            "Memory Orchestrator"
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show decisions without "
            "changing VM memory"
        )
    )

    args = parser.parse_args()

    orchestrator = (
        CentralMemoryOrchestrator(
            dry_run=args.dry_run
        )
    )

    orchestrator.run()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


