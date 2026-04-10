"""Phase 7: generate final end-to-end Python pipeline script artifact."""

from __future__ import annotations

import re
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState

# Libraries known to be in the standard AutoML stack.
# Imports of anything NOT in this set will be replaced in post-processing.
_ALLOWED_IMPORT_PREFIXES = (
    "pandas", "numpy", "sklearn", "scipy", "xgboost", "lightgbm",
    "catboost", "matplotlib", "seaborn", "joblib", "os", "sys",
    "csv", "json", "math", "typing", "pathlib", "warnings",
    "functools", "itertools", "collections", "datetime", "time",
    "io", "re", "copy", "pickle", "tempfile", "shutil",
)

_UNKNOWN_IMPORT_REPLACEMENT = (
    "# NOTE: import removed by post-processor (not in known stack). "
    "Use pandas/sklearn equivalents instead."
)


def _strip_unknown_imports(code: str) -> str:
    """Replace any import statement referencing a library not in the allowed set."""
    lines: list[str] = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            parts = stripped.split()
            # Determine the top-level module name
            if stripped.startswith("from "):
                # from X import Y  →  module = X
                module = parts[1].split(".")[0] if len(parts) > 1 else ""
            else:
                # import X  or  import X as Y  →  module = X
                module = parts[1].split(".")[0] if len(parts) > 1 else ""
            if module and not any(module == p or module.startswith(p) for p in _ALLOWED_IMPORT_PREFIXES):
                lines.append(f"# {_UNKNOWN_IMPORT_REPLACEMENT} (was: {line.strip()})")
                continue
        lines.append(line)
    return "\n".join(lines)


def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
    cols = profile.get("columns") or {}
    return {
        "n_rows": profile.get("n_rows"),
        "n_columns": profile.get("n_columns"),
        "columns_sample": dict(list(cols.items())[:30]),
    }


def _compact_decisions_for_code(entries: list[dict[str, Any]], limit: int = 25) -> str:
    """Returns a numbered list of decisions for embedding in the prompt."""
    lines: list[str] = []
    for i, e in enumerate(entries[-limit:], start=max(1, len(entries) - limit + 1)):
        step = e.get("step") or e.get("phase") or "?"
        action = e.get("action") or "?"
        rationale = str(e.get("rationale") or e.get("note") or "")[:180]
        lines.append(f"  Entry {i}: step={step} | action={action} | rationale={rationale}")
    return "\n".join(lines)


def _compact_exec_results(results: dict[str, Any]) -> dict[str, Any]:
    table = []
    for row in (results.get("comparison_table") or [])[:8]:
        table.append(
            {
                "model_name": row.get("model_name"),
                "cv_score": row.get("cv_score"),
                "test_score": row.get("test_score"),
                "status": row.get("status"),
            }
        )
    return {
        "best_model_name": results.get("best_model_name"),
        "metric": results.get("metric"),
        "table": table,
    }


def generate_pipeline_script(session: SessionState) -> str:
    """
    Phase 7 Step 3: produce one coherent, commented Python script.

    Every non-obvious line must include a comment referencing the log entry number
    that motivated it. Unknown imports are removed by post-processing.
    """
    decisions = episodic.get_log(session.decisions_log_id) if session.decisions_log_id else []
    latest = session.execution_results or {}
    decisions_block = _compact_decisions_for_code(decisions, limit=25)

    prompt = f"""Phase 7 Step 3 — Consolidated Code Output.

Generate a single, end-to-end Python script for this AutoML run. The script must be:
- Executable with standard libraries (pandas, sklearn, xgboost, lightgbm, scipy, numpy only)
- Fully commented: every non-obvious line must have an inline comment that references
  the decisions log entry that motivated it, using the format:
  # DecLog EntryN: <brief reason>
  For example:
  # DecLog Entry7: Cabin dropped — 77% null rate, binary has_cabin flag created instead
  df['has_cabin'] = df['Cabin'].notna().astype(int)
  df.drop('Cabin', axis=1, inplace=True)
- Structured as: (1) imports, (2) data loading, (3) preprocessing, (4) training, (5) prediction

CONTEXT:
TASK_CONFIG: {session.task_config}
DATASET_PROFILE: {_compact_profile(session.dataset_profile or {})}
PREPROCESSING_PLAN: {session.preprocessing_plan}
EXECUTION_PLAN: {session.execution_plan}
EXECUTION_RESULTS: {_compact_exec_results(latest)}

DECISIONS_LOG (reference entry numbers in your comments):
{decisions_block}

Critical rules:
- Only use imports from: pandas, numpy, sklearn, scipy, xgboost, lightgbm, warnings, os, pathlib
- No markdown fences, no explanatory prose — return Python code ONLY
- Include DATA_PATH and TARGET_COLUMN as top-level constants so users can swap them
- The script must produce a trained model and print predictions or a performance summary"""

    code = LLMClient().call(
        prompt,
        max_tokens=config.LLM_MAX_TOKENS_CODE,
        temperature=config.LLM_TEMPERATURE_JSON,
    ).strip()

    # Strip any markdown code fences the LLM may have added
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    if code:
        return _strip_unknown_imports(code)

    # Deterministic fallback script (with log-comment style)
    task = session.task_config or {}
    target_val = (task.get("target_column") or "target")
    best_model = (latest.get("best_model_name") or "RandomForestClassifier")
    est_import = (
        "from sklearn.linear_model import LogisticRegression"
        if "Logistic" in best_model
        else "from sklearn.ensemble import GradientBoostingClassifier"
        if "GradientBoosting" in best_model
        else "from sklearn.ensemble import RandomForestClassifier"
    )
    est_call = (
        "LogisticRegression(max_iter=1000, random_state=42)"
        if "Logistic" in best_model
        else "GradientBoostingClassifier(random_state=42)"
        if "GradientBoosting" in best_model
        else "RandomForestClassifier(n_estimators=100, random_state=42)"
    )
    return (
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        f"{est_import}\n"
        "from sklearn.metrics import classification_report\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# --- Configuration --- update to match your run\n"
        "DATA_PATH = 'your_dataset.csv'\n"
        f"TARGET_COLUMN = {repr(target_val)}\n"
        "\n"
        "df = pd.read_csv(DATA_PATH)\n"
        "\n"
        "# DecLog: minimal preprocessing fallback (see decisions log for full strategy)\n"
        "df = df.dropna(thresh=int(len(df.columns) * 0.5))\n"
        "X = df.drop(columns=[TARGET_COLUMN])\n"
        "y = df[TARGET_COLUMN]\n"
        "X = pd.get_dummies(X, dummy_na=True)\n"
        "X = X.fillna(X.median(numeric_only=True))\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 20 else None\n"
        ")\n"
        "\n"
        f"# DecLog: best model from Phase 5 = {best_model}\n"
        f"model = {est_call}\n"
        "model.fit(X_train, y_train)\n"
        "preds = model.predict(X_test)\n"
        "print(classification_report(y_test, preds))\n"
    )
