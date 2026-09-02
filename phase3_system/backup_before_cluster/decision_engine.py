"""
Turns a model prediction into a concrete action, applying:
  - probability thresholds (from your Phase 2 PR-curve tuning)
  - confidence gating (low confidence -> no-op, don't guess)
  - rate limiting (don't thrash the balloon)
  - hard memory floor/ceiling
"""

import time

import config


ACTION_INCREASE = "increase"
ACTION_DECREASE = "decrease"
ACTION_NONE = "none"


class DecisionEngine:
    def __init__(self):
        self._last_action_time = 0.0

    def _rate_limited(self):
        return (time.time() - self._last_action_time) < config.MIN_SECONDS_BETWEEN_ACTIONS

    def decide(self, pred_class, proba, current_memory_kib):
        """
        pred_class: raw class label from the model (e.g. 0/1, or "pressure"/"ok" —
                    adapt the comparisons below to your actual label encoding)
        proba: dict of class -> probability, or None
        current_memory_kib: current balloon target, for guardrail checks

        Returns: (action, reason_string, target_delta_kib)
        """
        # --- Confidence gate ---
        if proba is not None:
            max_conf = max(proba.values())
            if max_conf < config.MIN_CONFIDENCE_TO_ACT:
                return ACTION_NONE, f"low confidence ({max_conf:.2f})", 0

        # --- Rate limiting ---
        if self._rate_limited():
            return ACTION_NONE, "rate limited", 0

        # --- Threshold logic ---
        # Adapt these key lookups to your actual class labels from Phase 2.
        # Example assumes binary labels where "1" means "memory pressure expected".
        increase_proba = proba.get("1", 0.0) if proba else float(pred_class == 1)
        decrease_proba = 1.0 - increase_proba

        action = ACTION_NONE
        reason = "within normal range"
        delta = 0

        if increase_proba >= config.INCREASE_PROBA_THRESHOLD:
            if current_memory_kib + config.INCREMENT_KIB <= config.MAX_MEMORY_KIB:
                action = ACTION_INCREASE
                reason = f"increase_proba={increase_proba:.2f} >= threshold"
                delta = config.INCREMENT_KIB
            else:
                reason = "would exceed MAX_MEMORY_KIB guardrail — skipping"

        elif decrease_proba >= (1 - config.DECREASE_PROBA_THRESHOLD):
            if current_memory_kib - config.DECREMENT_KIB >= config.MIN_MEMORY_KIB:
                action = ACTION_DECREASE
                reason = f"decrease_proba={decrease_proba:.2f} >= threshold"
                delta = -config.DECREMENT_KIB
            else:
                reason = "would go below MIN_MEMORY_KIB guardrail — skipping"

        if action != ACTION_NONE:
            self._last_action_time = time.time()

        return action, reason, delta
