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
