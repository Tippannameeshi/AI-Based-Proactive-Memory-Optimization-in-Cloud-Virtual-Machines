"""
Main Phase 3 control loop:

    poll
      ->
    collect history
      ->
    build features
      ->
    ML prediction
      ->
    decision engine
      ->
    actuator
      ->
    feedback logging

Usage:
    python3 orchestrator.py --domain testvm --dry-run
    python3 orchestrator.py --domain testvm
    python3 orchestrator.py --domain testvm --dry-run --interval 10
"""

import argparse
import time
import traceback

import config
from collector import MemoryStatsCollector
from features import build_feature_vector
from model_wrapper import ModelWrapper
from decision_engine import DecisionEngine
from actuator import Actuator
from feedback_logger import FeedbackLogger


def run(
    domain_name,
    dry_run=False,
    poll_interval=config.POLL_INTERVAL_SECONDS
):
    print("\n================================================")
    print("     PHASE 3 VM MEMORY MANAGEMENT SYSTEM")
    print("================================================")

    print(f"\n[orchestrator] Domain: {domain_name}")
    print(f"[orchestrator] Dry run: {dry_run}")
    print(f"[orchestrator] Poll interval: {poll_interval}s")
    print(
        f"[orchestrator] History window: "
        f"{config.HISTORY_WINDOW_SIZE} samples"
    )

    print("\n[orchestrator] Initializing components...")

    collector = None

    try:
        # ------------------------------------------
        # INITIALIZATION
        # ------------------------------------------

        collector = MemoryStatsCollector(domain_name)

        model = ModelWrapper()

        engine = DecisionEngine()

        actuator = Actuator(
            collector.dom,
            dry_run=dry_run
        )

        logger = FeedbackLogger()

        print("[orchestrator] All components initialized successfully.")

        warmup_seconds = (
            config.HISTORY_WINDOW_SIZE * poll_interval
        )

        print(
            f"\n[orchestrator] Warming up..."
        )

        print(
            f"[orchestrator] Need "
            f"{config.HISTORY_WINDOW_SIZE} samples "
            f"(approximately {warmup_seconds}s)"
        )

        print(
            "\nPress Ctrl+C to stop.\n"
        )

        cycle = 0

        # ==========================================
        # MAIN CONTROL LOOP
        # ==========================================

        while True:

            cycle += 1

            try:
                # ----------------------------------
                # STEP 1: COLLECT LIVE METRICS
                # ----------------------------------

                sample = collector.poll()

                cpu_usage = sample.get(
                    "cpu_usage",
                    0.0
                )

                memory_usage = sample.get(
                    "memory_usage",
                    0.0
                )

                current_mem = sample.get(
                    "_current_memory_kib",
                    0
                )

                history_count = len(
                    collector.history
                )

                # ----------------------------------
                # WARMUP
                # ----------------------------------

                if not collector.is_ready():

                    print(
                        f"[WARMUP] "
                        f"{history_count}/"
                        f"{config.HISTORY_WINDOW_SIZE} | "
                        f"CPU={cpu_usage:.2f}% | "
                        f"MEM={memory_usage:.2f}%"
                    )

                    time.sleep(poll_interval)
                    continue

                # ----------------------------------
                # STEP 2: BUILD FEATURES
                # ----------------------------------

                history = collector.get_history()

                feature_vector = build_feature_vector(
                    history
                )

                # ----------------------------------
                # STEP 3: MODEL PREDICTION
                # ----------------------------------

                pred_class, probabilities = (
                    model.predict(feature_vector)
                )

                # ----------------------------------
                # STEP 4: DECISION ENGINE
                # ----------------------------------

                action, reason, delta = (
                    engine.decide(
                        pred_class,
                        probabilities,
                        current_mem,
                        sample["memory_usage"]
                    )
                )

                # ----------------------------------
                # STEP 5: ACTUATOR
                # ----------------------------------

                success, new_mem, message = (
                    actuator.apply(
                        action,
                        delta,
                        current_mem
                    )
                )

                # ----------------------------------
                # DISPLAY RESULT
                # ----------------------------------

                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                print("\n----------------------------------------")

                print(
                    f"[CYCLE {cycle}] {timestamp}"
                )

                print(
                    f"CPU Usage:    {cpu_usage:.2f}%"
                )

                print(
                    f"Memory Usage: {memory_usage:.2f}%"
                )

                print(
                    f"Current Memory: "
                    f"{current_mem / 1024:.0f} MB"
                )

                print(
                    f"Prediction: {pred_class}"
                )

                print(
                    f"Probabilities: {probabilities}"
                )

                print(
                    f"Decision: {action.upper()}"
                )

                print(
                    f"Reason: {reason}"
                )

                print(
                    f"Memory Delta: "
                    f"{delta / 1024:.0f} MB"
                )

                print(
                    f"Target Memory: "
                    f"{new_mem / 1024:.0f} MB"
                )

                print(
                    f"Actuator: {message}"
                )

                print(
                    f"Success: {success}"
                )

                print("----------------------------------------")

                # ----------------------------------
                # STEP 6: FEEDBACK LOGGING
                # ----------------------------------

                logger.log(
                    domain_name,
                    feature_vector,
                    pred_class,
                    probabilities,
                    action,
                    reason,
                    current_mem,
                    new_mem,
                    success
                )

            except Exception as e:

                print(
                    "\n[orchestrator] ERROR during cycle:"
                )

                print(
                    f"{type(e).__name__}: {e}"
                )

                traceback.print_exc()

            # --------------------------------------
            # WAIT FOR NEXT POLL
            # --------------------------------------

            time.sleep(poll_interval)

    except KeyboardInterrupt:

        print(
            "\n\n[orchestrator] "
            "Stopped safely by user."
        )

    finally:

        if collector is not None:
            collector.close()

        print(
            "[orchestrator] Libvirt connection closed."
        )


# ==============================================
# COMMAND LINE
# ==============================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 Proactive VM "
            "Memory Management System"
        )
    )

    parser.add_argument(
        "--domain",
        required=True,
        help="Libvirt domain name"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the complete pipeline "
            "without actually changing VM memory"
        )
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=config.POLL_INTERVAL_SECONDS,
        help="Polling interval in seconds"
    )

    args = parser.parse_args()

    run(
        domain_name=args.domain,
        dry_run=args.dry_run,
        poll_interval=args.interval
    )
