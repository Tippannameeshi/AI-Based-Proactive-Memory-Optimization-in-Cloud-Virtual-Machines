"""
Turns a model prediction into a concrete memory management action.

Model labels:
    1 -> memory pressure predicted -> increase VM memory
    0 -> normal / no pressure -> potentially decrease VM memory

Safety mechanisms:
    - confidence gating
    - probability thresholds
    - live memory usage safety checks
    - emergency high-memory protection
    - rate limiting
    - minimum/maximum memory guardrails
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
        """Prevent frequent VM memory resizing."""
        elapsed = time.time() - self._last_action_time
        return elapsed < config.MIN_SECONDS_BETWEEN_ACTIONS

    def decide(
        self,
        pred_class,
        proba,
        current_memory_kib,
        memory_usage
    ):
        """
        Decide whether VM memory should be increased,
        decreased, or left unchanged.

        Parameters
        ----------
        pred_class : int
            Model predicted class.
            1 = memory pressure
            0 = normal

        proba : dict
            Example:
            {0: 0.85, 1: 0.15}

        current_memory_kib : int
            Current VM allocated memory in KiB.

        memory_usage : float
            Actual live VM memory usage percentage.

        Returns
        -------
        tuple
            (action, reason, delta_kib)
        """

        # ==================================================
        # VALIDATE PROBABILITIES
        # ==================================================

        if proba is None:

            pressure_proba = (
                1.0 if int(pred_class) == 1 else 0.0
            )

            normal_proba = 1.0 - pressure_proba

        else:

            pressure_proba = float(
                proba.get(1, 0.0)
            )

            normal_proba = float(
                proba.get(0, 0.0)
            )

        # ==================================================
        # CONFIDENCE GATE
        # ==================================================

        max_confidence = max(
            pressure_proba,
            normal_proba
        )

        if max_confidence < config.MIN_CONFIDENCE_TO_ACT:

            return (
                ACTION_NONE,
                (
                    f"low model confidence "
                    f"({max_confidence:.3f})"
                ),
                0
            )

        # ==================================================
        # RATE LIMITING
        # ==================================================

        if self._rate_limited():

            remaining = (
                config.MIN_SECONDS_BETWEEN_ACTIONS
                - (
                    time.time()
                    - self._last_action_time
                )
            )

            return (
                ACTION_NONE,
                (
                    f"rate limited "
                    f"({remaining:.1f}s remaining)"
                ),
                0
            )

        # ==================================================
        # EMERGENCY HIGH MEMORY USAGE
        # ==================================================
        #
        # Even if ML model says normal, protect the VM
        # if actual memory usage becomes dangerously high.
        # ==================================================

        if (
            memory_usage
            >= config.HIGH_MEMORY_USAGE_THRESHOLD
        ):

            target_memory = (
                current_memory_kib
                + config.INCREMENT_KIB
            )

            if target_memory <= config.MAX_MEMORY_KIB:

                self._last_action_time = time.time()

                return (
                    ACTION_INCREASE,
                    (
                        f"emergency memory protection: "
                        f"usage {memory_usage:.2f}% >= "
                        f"{config.HIGH_MEMORY_USAGE_THRESHOLD}%"
                    ),
                    config.INCREMENT_KIB
                )

            return (
                ACTION_NONE,
                (
                    "increase blocked: "
                    "MAX_MEMORY_KIB guardrail reached"
                ),
                0
            )

        # ==================================================
        # MODEL PREDICTS MEMORY PRESSURE
        # ==================================================

        if (
            pressure_proba
            >= config.INCREASE_PROBA_THRESHOLD
        ):

            target_memory = (
                current_memory_kib
                + config.INCREMENT_KIB
            )

            if target_memory <= config.MAX_MEMORY_KIB:

                self._last_action_time = time.time()

                return (
                    ACTION_INCREASE,
                    (
                        f"memory pressure probability "
                        f"{pressure_proba:.3f} >= "
                        f"{config.INCREASE_PROBA_THRESHOLD}"
                    ),
                    config.INCREMENT_KIB
                )

            return (
                ACTION_NONE,
                (
                    "increase blocked: "
                    "MAX_MEMORY_KIB guardrail reached"
                ),
                0
            )

        # ==================================================
        # MODEL PREDICTS NORMAL
        # ==================================================
        #
        # Only decrease if:
        # 1. Model strongly predicts normal
        # 2. Actual live memory usage is sufficiently low
        # ==================================================

        if (
            normal_proba
            >= config.DECREASE_PROBA_THRESHOLD
        ):

            # Safety check before reclaiming memory
            if (
                memory_usage
                >= config.SAFE_MEMORY_USAGE_FOR_DECREASE
            ):

                return (
                    ACTION_NONE,
                    (
                        f"decrease blocked: live memory usage "
                        f"{memory_usage:.2f}% >= safe threshold "
                        f"{config.SAFE_MEMORY_USAGE_FOR_DECREASE}%"
                    ),
                    0
                )

            target_memory = (
                current_memory_kib
                - config.DECREMENT_KIB
            )

            if target_memory >= config.MIN_MEMORY_KIB:

                self._last_action_time = time.time()

                return (
                    ACTION_DECREASE,
                    (
                        f"normal probability "
                        f"{normal_proba:.3f} >= "
                        f"{config.DECREASE_PROBA_THRESHOLD} "
                        f"and memory usage "
                        f"{memory_usage:.2f}% < "
                        f"{config.SAFE_MEMORY_USAGE_FOR_DECREASE}%"
                    ),
                    -config.DECREMENT_KIB
                )

            return (
                ACTION_NONE,
                (
                    "decrease blocked: "
                    "MIN_MEMORY_KIB guardrail reached"
                ),
                0
            )

        # ==================================================
        # DEFAULT
        # ==================================================

        return (
            ACTION_NONE,
            (
                f"no threshold met "
                f"(pressure={pressure_proba:.3f}, "
                f"normal={normal_proba:.3f}, "
                f"memory_usage={memory_usage:.2f}%)"
            ),
            0
        )
