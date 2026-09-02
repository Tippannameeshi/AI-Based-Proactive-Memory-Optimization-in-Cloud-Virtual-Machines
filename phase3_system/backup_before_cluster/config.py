"""
Central configuration for the Phase 3 proactive memory management system.
Tune these values based on your VM sizing and your Phase 2 model's behavior.
"""

# --- libvirt connection ---
LIBVIRT_URI = "qemu:///system"

# --- Polling ---
POLL_INTERVAL_SECONDS = 5          # how often to pull memory stats
HISTORY_WINDOW_SIZE = 12           # number of past samples kept for feature building
                                    # (12 * 5s = 60s rolling window)

# --- Memory limits (KiB) — SAFETY GUARDRAILS ---
# Never let the balloon push a VM below/above these values.
MIN_MEMORY_KIB = 512 * 1024        # 512 MB floor
MAX_MEMORY_KIB = 4096 * 1024       # 4 GB ceiling (must be <= VM's max configured memory)

# --- Actuation step size ---
INCREMENT_KIB = 256 * 1024         # +256 MB per "increase" action
DECREMENT_KIB = 256 * 1024         # -256 MB per "decrease" action

# --- Rate limiting ---
MIN_SECONDS_BETWEEN_ACTIONS = 20   # don't resize more often than this (prevents thrashing)

# --- Model decision thresholds ---
# Your Phase 2 PR-curve work should have produced an optimal probability threshold —
# put it here instead of using the sklearn default of 0.5.
INCREASE_PROBA_THRESHOLD = 0.60
DECREASE_PROBA_THRESHOLD = 0.25    # below this, it's safe to reclaim memory
MIN_CONFIDENCE_TO_ACT = 0.55       # if max class proba is below this, fall back to no-op

# --- Model / scaler paths ---
MODEL_PATH = "models/best_model_improved.pkl"
SCALER_PATH = None

# --- Logging ---
FEEDBACK_LOG_PATH = "logs/feedback_log.csv"
ORCHESTRATOR_LOG_PATH = "logs/orchestrator.log"
