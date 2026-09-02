"""
Logs every polling cycle (features, prediction, decision, outcome) to a CSV.
This is what feeds your Phase 4 report and any future retraining.
"""

import csv
import os
import time

import config
from features import FEATURE_NAMES


class FeedbackLogger:
    def __init__(self, path=config.FEEDBACK_LOG_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                header = (
                    ["timestamp", "domain"]
                    + FEATURE_NAMES
                    + ["pred_class", "proba", "action", "reason",
                       "current_memory_kib", "new_memory_kib", "success"]
                )
                writer.writerow(header)

    def log(self, domain, feature_vector, pred_class, proba, action, reason,
             current_memory_kib, new_memory_kib, success):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            row = (
                [time.time(), domain]
                + list(feature_vector)
                + [pred_class, proba, action, reason,
                   current_memory_kib, new_memory_kib, success]
            )
            writer.writerow(row)
