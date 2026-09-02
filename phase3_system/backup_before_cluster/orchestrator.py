"""
Main control loop: poll -> build features -> predict -> decide -> actuate -> log.

Usage:
    python3 orchestrator.py --domain testvm
    python3 orchestrator.py --domain testvm --dry-run
    python3 orchestrator.py --domain testvm --interval 10
"""

import argparse
import time
import sys

import config
from collector import MemoryStatsCollector
from features import build_feature_vector
from model_wrapper import ModelWrapper
from decision_engine import DecisionEngine
from actuator import Actuator
from feedback_logger import FeedbackLogger


def run(domain_name, dry_run=False, poll_interval=config.POLL_INTERVAL_SECONDS):
    print(f"[orchestrator] Connecting to domain '{domain_name}' "
          f"(dry_run={dry_run})...")

    collector = MemoryStatsCollector(domain_name)
    model = ModelWrapper()
    engine = DecisionEngine()
    actuator = Actuator(collector.dom, dry_run=dry_run)
    logger = FeedbackLogger()

    print(f"[orchestrator] Warming up — need {config.HISTORY_WINDOW_SIZE} "
          f"samples before predictions start ({config.HISTORY_WINDOW_SIZE * poll_interval}s)...")

    try:
        while True:
            sample = collector.poll()

            if not collector.is_ready():
                print(f"[orchestrator] warming up "
                      f"({len(collector.history)}/{config.HISTORY_WINDOW_SIZE})")
                time.sleep(poll_interval)
                continue

            history = collector.get_history()
            fvec = build_feature_vector(history)

            pred_class, proba = model.predict(fvec)

            current_mem = sample["_current_memory_kib"]
            action, reason, delta = engine.decide(pred_class, proba, current_mem)

            success, new_mem, msg = actuator.apply(action, delta, current_mem)

            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"pred={pred_class} proba={proba} action={action} "
                  f"reason='{reason}' current={current_mem}KiB -> {new_mem}KiB | {msg}")

            logger.log(domain_name, fvec, pred_class, proba, action, reason,
                       current_mem, new_mem, success)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[orchestrator] Stopped by user.")
    finally:
        collector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="libvirt domain name")
    parser.add_argument("--dry-run", action="store_true",
                         help="log decisions but don't actually call setMemory()")
    parser.add_argument("--interval", type=int, default=config.POLL_INTERVAL_SECONDS,
                         help="polling interval in seconds")
    args = parser.parse_args()

    run(args.domain, dry_run=args.dry_run, poll_interval=args.interval)
