"""Phase 7: generate final end-to-end Python pipeline script artifact."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState

logger = logging.getLogger(__name__)

# Libraries known to be in the standard AutoML stack (and in the sandbox image).
# Imports of anything NOT in this set will be replaced in post-processing.
_ALLOWED_IMPORT_PREFIXES = (
    "pandas", "numpy", "sklearn", "scipy", "xgboost", "lightgbm",
    "catboost", "imblearn", "category_encoders", "matplotlib", "seaborn",
    "joblib", "os", "sys", "csv", "json", "math", "typing", "pathlib", "warnings",
    "functools", "itertools", "collections", "datetime", "time",
    "io", "re", "copy", "pickle", "tempfile", "shutil",
)

_UNKNOWN_IMPORT_REPLACEMENT = (
    "# NOTE: import removed by post-processor (not in known stack). "
    "Use pandas/sklearn equivalents instead."
)

# File names the script must use — the sandbox validator provides/reads exactly these.
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
PREDICTIONS_FILE = "predictions.csv"


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


def _strip_fences(code: str) -> str:
    """Remove markdown code fences the LLM may have added."""
    code = code.strip()
    if "```" in code:
        code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("```"))
    return code.strip()


def raw_schema(df: pd.DataFrame | None, limit: int = 60) -> str:
    """Exact columns of train.csv as the script will read them — the LLM must not
    guess names from the post-preprocessing profile (one-hot columns etc.)."""
    if df is None:
        return "(unavailable)"
    lines = []
    for c in list(df.columns)[:limit]:
        s = df[c]
        examples = [str(v)[:20] for v in s.dropna().unique()[:3]]
        lines.append(f"  {c!r}: dtype={s.dtype}, missing={int(s.isna().sum())}, examples={examples}")
    if df.shape[1] > limit:
        lines.append(f"  ... {df.shape[1] - limit} more columns")
    return f"{len(df)} rows\n" + "\n".join(lines)


def recipe_summary(session: SessionState) -> dict[str, Any]:
    """What each step actually did — lets the LLM reproduce the run instead of guessing."""
    results = session.execution_results or {}
    rp = results.get("replay") or {}
    return {
        "feature_selection_kept_columns": (session.preprocessing_meta or {}).get("feature_selection_columns"),
        "deep_cleaning_assignments": session.deep_cleaning_assignments,
        "engineered_features": [
            e.get("spec") for e in (session.feature_engineering_log or []) if e.get("status") == "added"
        ],
        "smote": rp.get("smote"),
        "log_target": rp.get("log_target"),
        "final_feature_columns": rp.get("feature_columns"),
        "model": rp.get("model"),
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


def best_model_spec(results: dict[str, Any]) -> dict[str, Any]:
    """Name + tuned hyperparameters of the best model, so the script can rebuild it exactly."""
    best_name = str(results.get("best_model_name") or "")
    row = next(
        (r for r in results.get("comparison_table") or [] if r.get("model_name") == best_name),
        {},
    )
    params = {
        k: v for k, v in (row.get("best_params") or {}).items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }
    return {"model_name": best_name, "hyperparameters": params}


def _library_versions() -> str:
    """Installed versions (the sandbox image pins the same ones), so the LLM uses current APIs."""
    from importlib.metadata import PackageNotFoundError, version

    out = []
    for pkg in ("scikit-learn", "pandas", "numpy", "xgboost", "lightgbm", "catboost"):
        try:
            out.append(f"{pkg} {version(pkg)}")
        except PackageNotFoundError:
            continue
    return ", ".join(out)


def _contract_block(session: SessionState) -> str:
    """Rules the script must follow so the sandbox validator can run and score it."""
    task = session.task_config or {}
    results = session.execution_results or {}
    target = str(task.get("target_column") or "target")
    task_type = str(results.get("task_type") or task.get("task_type") or "classification")
    spec = best_model_spec(results)
    proba_rule = (
        '- Also add columns "proba_0", "proba_1", ... with predict_proba output, one per class, '
        "classes in sorted order of the training labels (skip if the model has no predict_proba).\n"
        if task_type == "classification" else ""
    )
    return f"""DATA CONTRACT — the script is executed automatically to verify it; follow exactly:
- Top-level constants: TRAIN_PATH = "{TRAIN_FILE}", TEST_PATH = "{TEST_FILE}",
  TARGET_COLUMN = {target!r}, PREDICTIONS_PATH = "{PREDICTIONS_FILE}"
- Both CSVs have the same raw columns (including TARGET_COLUMN) and are the exact
  train/test split the AutoML run used. Task type: {task_type}.
- Library versions: {_library_versions()}. Use APIs valid for these versions
  (e.g. OneHotEncoder(sparse_output=False), not sparse=False).
- Fit every preprocessing step (imputation, encoding, scaling, feature selection,
  engineered features) on TRAIN rows only, then apply it to TEST.
- Train ONE final model: {spec['model_name'] or 'the best model'} with these tuned
  hyperparameters: {spec['hyperparameters'] or 'library defaults'}.
  If the name is an ensemble (e.g. "Stacking(A+B)" or "Ensemble(A+B)"), rebuild it with
  sklearn StackingClassifier/VotingClassifier (or the Regressor versions) over those model families.
  If the hyperparameters include decision_threshold, predict the positive class when
  predict_proba[:, 1] >= decision_threshold. Ignore entries that are not estimator arguments.
- No hyperparameter search and no cross-validation — train once on all TRAIN rows.
- CPU only: never set device="cuda", device="gpu", tree_method="gpu_hist" or task_type="GPU".
- Predict EVERY TEST row in the original order (never drop or reorder test rows — impute instead).
- Write PREDICTIONS_PATH with a column "prediction" holding predictions in the ORIGINAL
  target labels/units (invert any label encoding or log transform of the target).
{proba_rule}- Finally print the test score."""


def generate_pipeline_script(
    session: SessionState,
    feature_columns: list[str] | None = None,
    df_raw_train: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """
    Phase 7 Step 3: produce one coherent, commented Python script.

    Returns (script, source) where source is "assembled" (deterministic replay of the
    recorded run — see script_builder.py), "llm" or "fallback". The LLM is only used
    when the run cannot be replayed exactly. Every script follows the data contract in
    `_contract_block`, so it can be executed and scored by `pipeline.script_validation`.
    """
    from automl_system.pipeline.script_builder import UnsupportedRecipe, assemble_pipeline_script

    try:
        return assemble_pipeline_script(session), "assembled"
    except UnsupportedRecipe as exc:
        logger.info("Script assembly not possible (%s) — generating with the LLM", exc)

    decisions = episodic.get_log(session.decisions_log_id) if session.decisions_log_id else []
    latest = session.execution_results or {}
    decisions_block = _compact_decisions_for_code(decisions, limit=25)
    features_line = (
        f"FINAL_FEATURES used by the best model (after preprocessing): {feature_columns[:60]}\n"
        if feature_columns else ""
    )

    prompt = f"""Phase 7 Step 3 — Consolidated Code Output.

Generate a single, end-to-end Python script for this AutoML run. The script must be:
- Executable with: pandas, numpy, sklearn, scipy, xgboost, lightgbm, catboost, imblearn
- Fully commented: every non-obvious line must have an inline comment that references
  the decisions log entry that motivated it, using the format:
  # DecLog EntryN: <brief reason>
  For example:
  # DecLog Entry7: Cabin dropped — 77% null rate, binary has_cabin flag created instead
  df['has_cabin'] = df['Cabin'].notna().astype(int)
  df.drop('Cabin', axis=1, inplace=True)
- Structured as: (1) imports, (2) data loading, (3) preprocessing, (4) training, (5) prediction

{_contract_block(session)}

CONTEXT:
TASK_CONFIG: {session.task_config}
RAW_TRAIN_SCHEMA (exact columns of train.csv — use ONLY these names; derived columns must be created by the script):
{raw_schema(df_raw_train)}
RECIPE (what the run actually did — reproduce it step by step):
{recipe_summary(session)}
PREPROCESSING_PLAN: {session.preprocessing_plan}
EXECUTION_PLAN: {session.execution_plan}
EXECUTION_RESULTS: {_compact_exec_results(latest)}
{features_line}
DECISIONS_LOG (reference entry numbers in your comments):
{decisions_block}

Critical rules:
- No markdown fences, no explanatory prose — return Python code ONLY"""

    code = _strip_fences(
        LLMClient().call(
            prompt,
            model=config.LLM_PRO,
            max_tokens=config.LLM_MAX_TOKENS_CODE,
            temperature=config.LLM_TEMPERATURE_JSON,
        )
    )
    if code:
        return _strip_unknown_imports(code), "llm"
    return fallback_pipeline_script(session), "fallback"


def repair_pipeline_script(
    session: SessionState, code: str, feedback: str, df_raw_train: pd.DataFrame | None = None
) -> str:
    """Ask the LLM to fix a script that failed validation. Returns "" on failure."""
    prompt = f"""The Python script below was checked automatically and did not pass validation.

PROBLEM:
{feedback[-4000:]}

{_contract_block(session)}

RAW_TRAIN_SCHEMA (exact columns of train.csv):
{raw_schema(df_raw_train)}

Fix the script so it runs and follows the contract. Keep the existing # DecLog comments.
Return the COMPLETE corrected script — Python code ONLY, no markdown fences, no prose.

SCRIPT:
{code}"""
    fixed = _strip_fences(
        LLMClient().call(
            prompt,
            model=config.LLM_PRO,
            max_tokens=config.LLM_MAX_TOKENS_CODE,
            temperature=config.LLM_TEMPERATURE_JSON,
        )
    )
    return _strip_unknown_imports(fixed) if fixed else ""


def fallback_pipeline_script(session: SessionState) -> str:
    """Deterministic script (with log-comment style) used when the LLM returns nothing."""
    task = session.task_config or {}
    results = session.execution_results or {}
    target_val = task.get("target_column") or "target"
    is_clf = str(results.get("task_type") or task.get("task_type") or "classification") == "classification"
    best_model = results.get("best_model_name") or "RandomForest"
    est = "RandomForestClassifier" if is_clf else "RandomForestRegressor"
    proba = (
        "if hasattr(model, 'predict_proba'):\n"
        "    for i, col in enumerate(model.predict_proba(X_test).T):\n"
        "        out[f'proba_{i}'] = col\n"
        if is_clf else ""
    )
    return (
        "import pandas as pd\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.pipeline import Pipeline, make_pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        f"from sklearn.ensemble import {est}\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# --- Configuration --- update to match your run\n"
        f"TRAIN_PATH = {TRAIN_FILE!r}\n"
        f"TEST_PATH = {TEST_FILE!r}\n"
        f"TARGET_COLUMN = {target_val!r}\n"
        f"PREDICTIONS_PATH = {PREDICTIONS_FILE!r}\n"
        "\n"
        "train = pd.read_csv(TRAIN_PATH)\n"
        "test = pd.read_csv(TEST_PATH)\n"
        "X_train, y_train = train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN]\n"
        "X_test = test.drop(columns=[TARGET_COLUMN], errors='ignore')\n"
        "\n"
        "# DecLog: minimal preprocessing fallback (see decisions log for full strategy)\n"
        "num_cols = X_train.select_dtypes('number').columns.tolist()\n"
        "cat_cols = [c for c in X_train.columns if c not in num_cols]\n"
        "prep = ColumnTransformer([\n"
        "    ('num', SimpleImputer(strategy='median'), num_cols),\n"
        "    ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),\n"
        "                          OneHotEncoder(handle_unknown='ignore')), cat_cols),\n"
        "])\n"
        "\n"
        f"# DecLog: best model from Phase 5 = {best_model} (fallback uses RandomForest)\n"
        f"model = Pipeline([('prep', prep), ('model', {est}(n_estimators=200, random_state=42))])\n"
        "model.fit(X_train, y_train)\n"
        "\n"
        "out = pd.DataFrame({'prediction': model.predict(X_test)})\n"
        f"{proba}"
        "out.to_csv(PREDICTIONS_PATH, index=False)\n"
        "if TARGET_COLUMN in test.columns:\n"
        "    print('test score:', model.score(X_test, test[TARGET_COLUMN]))\n"
    )
