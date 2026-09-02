"""
Central configuration for the Phase 3 proactive memory management system.
"""

# ============================================================
# LIBVIRT CONNECTION
# ============================================================

LIBVIRT_URI = "qemu:///system"


# ============================================================
# POLLING
# ============================================================

POLL_INTERVAL_SECONDS = 5

# 12 samples × 5 seconds = 60-second rolling history
HISTORY_WINDOW_SIZE = 12


# ============================================================
# MEMORY SAFETY GUARDRAILS
# ============================================================

# Never allow VM memory below 512 MB
MIN_MEMORY_KIB = 512 * 1024

# Maximum memory = 4 GB
# Must not exceed VM's configured maximum memory
MAX_MEMORY_KIB = 4096 * 1024


# ============================================================
# MEMORY RESIZE STEP
# ============================================================

# Increase VM memory by 256 MB
INCREMENT_KIB = 256 * 1024

# Decrease VM memory by 256 MB
DECREMENT_KIB = 256 * 1024


# ============================================================
# RATE LIMITING
# ============================================================

# Prevent frequent resizing / memory thrashing
MIN_SECONDS_BETWEEN_ACTIONS = 20


# ============================================================
# MODEL DECISION THRESHOLDS
# ============================================================

# Increase memory when pressure probability is sufficiently high
INCREASE_PROBA_THRESHOLD = 0.60

# Decrease memory when normal probability is sufficiently high
DECREASE_PROBA_THRESHOLD = 0.80

# Do nothing if model confidence is below this
MIN_CONFIDENCE_TO_ACT = 0.55


# ============================================================
# LIVE MEMORY SAFETY THRESHOLDS
# ============================================================

# Only reclaim/decrease allocated VM memory if actual memory
# usage is below this percentage.
SAFE_MEMORY_USAGE_FOR_DECREASE = 40.0

# Emergency safety threshold.
# If actual memory usage reaches this percentage, increase memory
# even if the ML model is uncertain.
HIGH_MEMORY_USAGE_THRESHOLD = 85.0


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = "models/xgb_model.pkl"

# Current XGBoost model does not require external scaling
SCALER_PATH = None


# ============================================================
# LOGGING
# ============================================================

FEEDBACK_LOG_PATH = "logs/feedback_log.csv"

ORCHESTRATOR_LOG_PATH = "logs/orchestrator.log"
