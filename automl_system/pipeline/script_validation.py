"""Phase 7: multi-stage validation of the generated pipeline script, with repair.

1. Static   — syntax, imports, keyword arguments vs installed library signatures,
              data contract (script_checks.static_check) — no execution needed.
2. Runtime  — run in the sandbox (no network, CPU/memory/time limits).
3. Semantic — predictions.csv has one row per test row, in original label space,
              and its score reproduces the reported test score within tolerance.
4. Feedback — any failure is sent back to the LLM with the exact error for repair.

The script's own printed score is never trusted — the host scores predictions.csv
with the same metric function Phase 5 used.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.pipeline.code_generator import (
    PREDICTIONS_FILE,
    TEST_FILE,
    TRAIN_FILE,
    repair_pipeline_script,
)
from automl_system.pipeline.execution import _test_score, _to_display_score
from automl_system.pipeline.script_checks import static_check
from automl_system.sandbox import start_sandbox
from automl_system.state.session import SessionState
from automl_system.utils.metric_labels import is_neg_metric

logger = logging.getLogger(__name__)

STATUS_VERIFIED = "verified"          # runs and reproduces the reported score
STATUS_MISMATCH = "score_mismatch"    # runs, but the score differs beyond tolerance
STATUS_FAILED = "failed"              # never ran successfully
STATUS_SKIPPED = "skipped"            # validation not possible (sandbox off, no data, ...)


def _norm_label(v: Any) -> str:
    """Compare labels by value, so 1, 1.0 and "1" all match."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v).strip()


def _score_predictions(
    preds: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    metric: str,
    task_type: str,
) -> tuple[float | None, str | None]:
    """Return (raw score, None) or (None, problem description for the LLM)."""
    if "prediction" not in preds.columns:
        return None, f'{PREDICTIONS_FILE} has no "prediction" column (columns: {list(preds.columns)[:10]}).'
    if len(preds) != len(df_test):
        return None, (
            f"{PREDICTIONS_FILE} has {len(preds)} rows but the test set has {len(df_test)}. "
            "Predict every test row in order — impute missing values instead of dropping rows."
        )

    if task_type == "classification":
        # Same encoding Phase 5 used: LabelEncoder over sorted training labels.
        classes = [_norm_label(v) for v in np.unique(df_train[target].dropna())]
        index = {c: i for i, c in enumerate(classes)}
        y_true = df_test[target].map(lambda v: index.get(_norm_label(v), -1)).to_numpy()
        y_pred = preds["prediction"].map(lambda v: index.get(_norm_label(v), -1)).to_numpy()
        unknown = float(np.mean(y_pred == -1))
        if unknown > 0.01:
            sample = preds["prediction"].head(5).tolist()
            return None, (
                f"{unknown:.0%} of predictions are not original target labels "
                f"(expected values from {classes[:10]}, got e.g. {sample}). "
                "Invert any label encoding before writing predictions."
            )
        proba_cols = [f"proba_{i}" for i in range(len(classes))]
        y_proba = (
            preds[proba_cols].to_numpy(dtype=float)
            if all(c in preds.columns for c in proba_cols) else None
        )
    else:
        y_true = pd.to_numeric(df_test[target], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(preds["prediction"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            return None, "Predictions contain NaN, inf or non-numeric values."
        y_proba = None

    try:
        return float(_test_score(metric, task_type, y_true, y_pred, y_proba)), None
    except Exception as exc:
        return None, f"Could not score predictions with metric {metric!r}: {exc}"


def _within_tolerance(reported: float, reproduced: float, metric: str) -> bool:
    gap = abs(reported - reproduced)
    if is_neg_metric(metric):  # error metrics (RMSE, MAE): scale-dependent → relative
        return gap <= config.SCRIPT_VALIDATION_REL_TOLERANCE * max(abs(reported), 1e-9)
    return gap <= config.SCRIPT_VALIDATION_TOLERANCE


def validate_and_repair_script(
    session: SessionState,
    script: str,
    df_train: pd.DataFrame | None,
    df_test: pd.DataFrame | None,
    source: str = "llm",
) -> tuple[str, dict[str, Any]]:
    """
    Run `script` against the run's train/test split; repair up to
    SCRIPT_VALIDATION_MAX_REPAIRS times. Returns (script to deliver, validation report).
    `source` is how the script was produced ("assembled", "llm" or "fallback").

    Reads `session` but never mutates it (safe to run beside report generation).
    """
    results = session.execution_results or {}
    target = str((session.task_config or {}).get("target_column") or "")
    metric = str(results.get("metric") or "")
    task_type = str(results.get("task_type") or "")
    reported_raw = results.get("best_test_score")
    report: dict[str, Any] = {
        "status": STATUS_SKIPPED,
        "source": source,
        "first_attempt_ok": False,
        "metric": metric,
        "metric_display": results.get("metric_display") or metric,
        "best_model": results.get("best_model_name"),
        "reported_score": _to_display_score(reported_raw, metric) if reported_raw is not None else None,
        "reproduced_score": None,
        "gap": None,
        "attempts": 0,
        "backend": None,
        "duration_s": 0.0,
        "history": [],
        "message": "",
    }

    if not script.strip():
        report["message"] = "No script to validate."
        return script, report
    if df_train is None or df_test is None or not target or target not in df_test.columns:
        report["message"] = "Train/test split not available — script not executed."
        return script, report
    if not isinstance(reported_raw, (int, float)) or not metric:
        report["message"] = "No reported test score to compare against — script not executed."
        return script, report

    t0 = time.perf_counter()
    try:
        sandbox, fallback_note = start_sandbox()
        if sandbox is None:
            report["message"] = "Sandbox disabled (SANDBOX_BACKEND=off)."
            return script, report
        with sandbox as sb:
            report["backend"] = sb.backend
            sb.put_dataframe(TRAIN_FILE, df_train)
            sb.put_dataframe(TEST_FILE, df_test)
            final_script = _run_repair_loop(session, script, sb, df_train, df_test,
                                            target, metric, task_type, float(reported_raw), report)
        if fallback_note:
            report["message"] = f"{report['message']} ({fallback_note})"
    except Exception as exc:
        logger.exception("Script validation crashed")
        report["status"] = STATUS_SKIPPED
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        report["message"] = f"Script not validated — the sandbox could not run ({first_line[:160]})."
        final_script = script
    report["duration_s"] = round(time.perf_counter() - t0, 2)
    return final_script, report


def _run_repair_loop(
    session: SessionState,
    script: str,
    sb: Any,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
    metric: str,
    task_type: str,
    reported_raw: float,
    report: dict[str, Any],
) -> str:
    code = script
    best_running: tuple[float, str, float] | None = None  # (gap, code, raw score)
    max_attempts = 1 + max(0, config.SCRIPT_VALIDATION_MAX_REPAIRS)

    for attempt in range(1, max_attempts + 1):
        score, problem = None, None
        issues = static_check(code)
        if issues:
            # Stage 1 failed — skip the sandbox run, go straight to repair.
            problem = "Static analysis (before execution) found:\n- " + "\n- ".join(issues[:15])
            entry: dict[str, Any] = {"attempt": attempt, "stage": "static", "exit_code": None,
                                     "duration_s": 0.0, "reproduced_score": None, "problem": problem[:500]}
            report["history"].append(entry)
            report["attempts"] = attempt
            if attempt == max_attempts:
                break
            repaired = repair_pipeline_script(session, code, problem, df_train)
            if not repaired or repaired.strip() == code.strip():
                entry["problem"] += " | repair produced no change"
                break
            code = repaired
            continue

        (sb.workdir / PREDICTIONS_FILE).unlink(missing_ok=True)
        res = sb.run(code)
        if res.timed_out:
            problem = (
                f"The script exceeded the {config.SANDBOX_TIMEOUT_SEC}s time limit. "
                "Remove any hyperparameter search or cross-validation and train a single model."
            )
        elif not res.ok:
            problem = "The script crashed:\n" + res.stderr[-3000:]
        else:
            text = sb.read_text(PREDICTIONS_FILE)
            if text is None:
                problem = f"The script finished but did not write {PREDICTIONS_FILE}."
            else:
                try:
                    preds = pd.read_csv(io.StringIO(text))
                    score, problem = _score_predictions(preds, df_train, df_test, target, metric, task_type)
                except Exception as exc:
                    problem = f"Could not read {PREDICTIONS_FILE}: {exc}"

        entry = {
            "attempt": attempt,
            "stage": "semantic" if res.ok else "runtime",
            "exit_code": res.exit_code,
            "duration_s": res.duration_s,
            "reproduced_score": _to_display_score(score, metric) if score is not None else None,
            "problem": (problem or "")[:500],
        }
        report["history"].append(entry)
        report["attempts"] = attempt

        if score is not None:
            gap = abs(float(_to_display_score(reported_raw, metric)) - float(_to_display_score(score, metric)))
            entry["gap"] = round(gap, 6)
            if best_running is None or gap < best_running[0]:
                best_running = (gap, code, score)
            if _within_tolerance(reported_raw, score, metric):
                report["first_attempt_ok"] = attempt == 1
                break
            problem = (
                f"The script runs, but its test {report['metric_display']} is "
                f"{_to_display_score(score, metric):.4f} while the AutoML run reported "
                f"{_to_display_score(reported_raw, metric):.4f}. Make the preprocessing, model "
                "and hyperparameters match the run exactly (see the decisions log comments and contract)."
            )

        if attempt == max_attempts:
            break
        repaired = repair_pipeline_script(session, code, problem or "", df_train)
        if not repaired or repaired.strip() == code.strip():
            entry["problem"] += " | repair produced no change"
            break
        code = repaired

    if best_running is None:
        report["status"] = STATUS_FAILED
        report["message"] = (
            f"Script did not run successfully after {report['attempts']} attempt(s): "
            f"{report['history'][-1]['problem'][:200]}"
        )
        return script  # repairs never ran either — deliver the original

    gap, best_code, best_score = best_running
    report["reproduced_score"] = _to_display_score(best_score, metric)
    report["gap"] = round(gap, 6)
    if _within_tolerance(reported_raw, best_score, metric):
        report["status"] = STATUS_VERIFIED
        report["message"] = (
            f"Script reproduces the reported {report['metric_display']} "
            f"({report['reproduced_score']:.4f} vs {report['reported_score']:.4f})."
        )
    else:
        report["status"] = STATUS_MISMATCH
        report["message"] = (
            f"Script runs, but its {report['metric_display']} is {report['reproduced_score']:.4f} "
            f"vs {report['reported_score']:.4f} reported (gap {gap:.4f})."
        )
    return best_code
