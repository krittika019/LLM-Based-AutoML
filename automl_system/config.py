"""Central configuration: loads .env then exposes constants (defaults + environment overrides)."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # optional; set env vars in the shell instead

    def load_dotenv(*_args, **_kwargs) -> None:
        return None


# Project root = parent of the automl_system package (where .env should live)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)
load_dotenv(override=True)  # current working directory (e.g. if launched from elsewhere)

# --- Secrets & deployment (override via .env) ---
# Groq: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Second Groq account/key, used only as the fallback when the primary is rate-limited or erroring.
GROQ_API_KEY_FALLBACK = os.getenv("GROQ_API_KEY_FALLBACK", "").strip()

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017").strip()
DATABASE_NAME = os.getenv("DATABASE_NAME", "automl_system").strip()
EPISODIC_COLLECTION = os.getenv("EPISODIC_COLLECTION", "episodic_memory").strip()


# Groq model ids (see https://console.groq.com/docs/models)
LLM_FLASH = os.getenv("LLM_FLASH", "openai/gpt-oss-20b").strip()
LLM_PRO = os.getenv("LLM_PRO", "openai/gpt-oss-120b").strip()

# ── Alternative LLM providers (all OpenAI-compatible) ──────────────────────
# Set LLM_PROVIDER to one of: groq | cerebras | together | ollama | openai
# Cerebras is the fastest free-tier option (https://inference.cerebras.ai/)
# Together AI: https://api.together.xyz  |  Ollama: http://localhost:11434
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
# Fallback provider tried once if primary exhausts all retries.
# "groq_fallback" = a second Groq account (GROQ_API_KEY_FALLBACK), same models as primary.
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "groq_fallback").strip().lower()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# Generic override: any OpenAI-compatible base URL (e.g. Ollama, vLLM, LM Studio)
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()

# Provider-specific model name overrides (sensible defaults per provider)
LLM_FLASH_CEREBRAS = os.getenv("LLM_FLASH_CEREBRAS", "qwen-3.8-27b").strip()
LLM_PRO_CEREBRAS = os.getenv("LLM_PRO_CEREBRAS", "gpt-oss-120b").strip()
LLM_FLASH_TOGETHER = os.getenv("LLM_FLASH_TOGETHER", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo").strip()
LLM_PRO_TOGETHER = os.getenv("LLM_PRO_TOGETHER", "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo").strip()
LLM_FLASH_OLLAMA = os.getenv("LLM_FLASH_OLLAMA", "llama3.2").strip()
LLM_PRO_OLLAMA = os.getenv("LLM_PRO_OLLAMA", "llama3.3").strip()
LLM_FLASH_OPENAI = os.getenv("LLM_FLASH_OPENAI", "gpt-4o-mini").strip()
LLM_PRO_OPENAI = os.getenv("LLM_PRO_OPENAI", "gpt-4o").strip()

# ── LLM call efficiency settings ───────────────────────────────────────────
# Caps keep responses concise and speed up generation significantly
LLM_MAX_TOKENS_JSON = int(os.getenv("LLM_MAX_TOKENS_JSON", "2048"))
LLM_MAX_TOKENS_NARRATIVE = int(os.getenv("LLM_MAX_TOKENS_NARRATIVE", "1024"))
LLM_MAX_TOKENS_CODE = int(os.getenv("LLM_MAX_TOKENS_CODE", "3500"))
# Lower temperature → more deterministic, shorter, faster for structured outputs
LLM_TEMPERATURE_JSON = float(os.getenv("LLM_TEMPERATURE_JSON", "0.0"))
LLM_TEMPERATURE_NARRATIVE = float(os.getenv("LLM_TEMPERATURE_NARRATIVE", "0.5"))

# Connection / operation behavior
MONGODB_CONNECT_TIMEOUT_MS = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000"))
MONGODB_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000"))
MAX_DB_RETRIES = int(os.getenv("MAX_DB_RETRIES", "3"))

# Per-step verification retries (orchestrator routes back to failed node)
MAX_STEP_RETRIES = int(os.getenv("MAX_STEP_RETRIES", "2"))

# AutoML framework benchmark time budget per framework (seconds).
# Set to 0 in .env to disable all benchmarks.
AUTOML_BENCHMARK_TIME_LIMIT = int(os.getenv("AUTOML_BENCHMARK_TIME_LIMIT", "120"))
# The web app skips those comparison runs unless this is true (see server.py).
APP_RUN_COMPETITORS: bool = os.getenv("APP_RUN_COMPETITORS", "false").strip().lower() in ("1", "true", "yes")

# Application limits (optional env overrides for ops tuning)
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "1000000"))
MAX_DATASET_COLUMNS = int(os.getenv("MAX_DATASET_COLUMNS", "10000"))

# LLM call retries (transient API errors)
MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "3"))
LLM_RETRY_BACKOFF_SEC = float(os.getenv("LLM_RETRY_BACKOFF_SEC", "1.0"))

# Leakage correlation threshold — features with |corr| >= this value with the target
# are flagged as potential leakage. Used consistently in feature_selection, deep_cleaning,
# and verification to avoid false-negatives from threshold mismatch.
LEAKAGE_CORR_THRESHOLD: float = float(os.getenv("LEAKAGE_CORR_THRESHOLD", "0.95"))

# Test scores at or above this value are flagged as potential data leakage and excluded
# from the best-model recommendation.
NEAR_PERFECT_SCORE_SENTINEL: float = float(os.getenv("NEAR_PERFECT_SCORE_SENTINEL", "0.999"))

# Memory ablation: set USE_MEMORY=false to disable episodic memory.
# Used in research experiments to measure the contribution of the decision log.
_use_mem_raw = os.getenv("USE_MEMORY", "true").strip().lower()
USE_MEMORY: bool = _use_mem_raw not in ("0", "false", "no")

# Heavy reasoning steps use LLM_PRO when wired by orchestration (not env — app logic)
HEAVY_REASONING_STEPS = ["eda_narrative", "failure_diagnosis", "report_generation"]

# ── Sandbox for executing generated code ───────────────────────────────────
# auto   = Docker if the sandbox image is available, otherwise local subprocess
# docker = Docker only (error if unavailable) | local = subprocess only | off = never execute
SANDBOX_BACKEND = os.getenv("SANDBOX_BACKEND", "auto").strip().lower()
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "automl-sandbox:1").strip()
SANDBOX_MEMORY = os.getenv("SANDBOX_MEMORY", "2g").strip()
SANDBOX_CPUS = float(os.getenv("SANDBOX_CPUS", "2"))
SANDBOX_TIMEOUT_SEC = int(os.getenv("SANDBOX_TIMEOUT_SEC", "300"))
# Max wait for Docker to start a container; on a busy laptop Docker's VM can stall,
# and "auto" then falls back to the local subprocess instead of skipping validation.
SANDBOX_START_TIMEOUT_SEC = int(os.getenv("SANDBOX_START_TIMEOUT_SEC", "45"))
# Start Docker Desktop only when a run needs the sandbox and stop it again afterwards
# (only if the app started it). Its VM holds 1-3 GB of RAM even when idle.
_on_demand_raw = os.getenv("SANDBOX_DOCKER_ON_DEMAND", "true").strip().lower()
SANDBOX_DOCKER_ON_DEMAND: bool = _on_demand_raw not in ("0", "false", "no")
SANDBOX_DOCKER_BOOT_TIMEOUT_SEC = int(os.getenv("SANDBOX_DOCKER_BOOT_TIMEOUT_SEC", "120"))
# Container self-destructs after this, even if the app crashes before cleanup
SANDBOX_MAX_LIFETIME_SEC = int(os.getenv("SANDBOX_MAX_LIFETIME_SEC", "7200"))

# Feature engineering: "cv" keeps an LLM-proposed feature only if it improves the
# cross-validated score of a fast LightGBM proxy (computed from original-scale
# columns); "corr" is the older correlation filter on deep-cleaned columns.
FE_SELECTION = os.getenv("FE_SELECTION", "cv").strip().lower()
FE_ROUNDS = int(os.getenv("FE_ROUNDS", "2"))               # propose → measure → feed back
FE_MIN_CV_GAIN = float(os.getenv("FE_MIN_CV_GAIN", "0.0005"))
# LLM-written feature functions (group statistics, frequencies, binning, text) executed
# in the Docker sandbox. "auto" = on when the Docker sandbox is available, "off" = never.
# Off by default: on titanic/heart/churn/bank no code feature passed the CV gate, and the
# longer prompts add LLM cost and latency. Worth enabling for text/high-cardinality data.
FE_CODE_FEATURES = os.getenv("FE_CODE_FEATURES", "off").strip().lower()
FE_CODE_TIMEOUT_SEC = int(os.getenv("FE_CODE_TIMEOUT_SEC", "120"))

# Phase 6 diagnosis: controlled CV experiments (probes) run before the LLM diagnoses a
# missed threshold. A probe counts as evidence when its paired gain exceeds this value
# and one standard error of the folds.
DIAG_PROBES = os.getenv("DIAG_PROBES", "true").strip().lower() not in ("0", "false", "no")
DIAG_MIN_GAIN = float(os.getenv("DIAG_MIN_GAIN", "0.002"))

# Phase 7 script validation: run the generated script in the sandbox and check it
# reproduces the reported test score. Gap tolerance is absolute for bounded metrics
# (accuracy, F1, R², AUC) and relative (×reported score) for error metrics (RMSE, MAE).
SCRIPT_VALIDATION_MAX_REPAIRS = int(os.getenv("SCRIPT_VALIDATION_MAX_REPAIRS", "3"))
SCRIPT_VALIDATION_TOLERANCE = float(os.getenv("SCRIPT_VALIDATION_TOLERANCE", "0.02"))
SCRIPT_VALIDATION_REL_TOLERANCE = float(os.getenv("SCRIPT_VALIDATION_REL_TOLERANCE", "0.05"))
