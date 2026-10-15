"""Phase 3 Step 5: LLM-proposed feature engineering with leakage guard."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.pipeline.code_features import (
    CodeFeatureRunner,
    check_feature_code,
    code_features_enabled,
    is_safety_problem,
    normalize_feature_code,
)
from automl_system.pipeline.replay import (
    add_raw_features,
    compute_raw_feature,
    finish_feature,
    safe_numeric_features,
)
from automl_system.state.session import SessionState
from automl_system.utils.dataset_profile import compute_dataset_profile
from automl_system.utils.decision_log import log_decision
from automl_system.utils.decision_router import RouterStats, should_engineer_features
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)

_LEAKAGE_CORR_THRESHOLD = 0.95
_MIN_USEFUL_CORR = 0.03
_MIN_ROWS_FOR_FE = 200       # skip entirely if fewer rows than this


def _dt_reasoning_block(
    df: pd.DataFrame,
    numeric: list[str],
    y: pd.Series,
    task_type: str,
) -> str:
    """Train a shallow decision tree and return a block of split rules for the LLM prompt.

    Grounds LLM feature proposals in actual data patterns (NeurIPS 2024, OCTree).
    Returns an empty string on any failure so the rest of FE still proceeds.
    """
    if len(numeric) < 2 or len(df) < 50:
        return ""
    try:
        from sklearn.preprocessing import LabelEncoder as _LE
        from sklearn.tree import DecisionTreeClassifier as _DTC, DecisionTreeRegressor as _DTR

        X_dt = df[numeric].fillna(df[numeric].median())
        y_dt = y.copy()

        if task_type == "classification" or not pd.api.types.is_numeric_dtype(y_dt) or y_dt.nunique() <= 15:
            if not pd.api.types.is_numeric_dtype(y_dt):
                y_dt = pd.Series(_LE().fit_transform(y_dt.astype(str)), index=y_dt.index)
            dt = _DTC(max_depth=4, random_state=42, min_samples_leaf=max(5, len(df) // 50))
        else:
            dt = _DTR(max_depth=4, random_state=42, min_samples_leaf=max(5, len(df) // 50))

        dt.fit(X_dt, y_dt)

        # Top features by impurity reduction
        importances = dt.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_features = [
            (numeric[i], round(float(importances[i]), 3))
            for i in top_idx if importances[i] > 0.01
        ]
        if not top_features:
            return ""

        # Key splitting rules from the first 2 tree levels
        tree = dt.tree_
        rules: list[str] = []

        def _extract(node: int, depth: int = 0) -> None:
            if depth >= 2 or len(rules) >= 5 or tree.feature[node] < 0:
                return
            fname = numeric[tree.feature[node]]
            thr = tree.threshold[node]
            n = int(tree.n_node_samples[node])
            rules.append(f"{fname} <= {thr:.3g}  ({n} samples split here)")
            _extract(tree.children_left[node], depth + 1)
            _extract(tree.children_right[node], depth + 1)

        _extract(0)

        lines = [
            "DT_REASONING (shallow decision tree on cleaned data - use this to ground your proposals):",
            f"  Most predictive features: {top_features}",
        ]
        if rules:
            lines.append(f"  Key decision boundaries: {rules}")
        lines.append(
            "  Propose features that sharpen these boundaries or reveal patterns the tree misses."
        )
        return "\n".join(lines)
    except Exception as _e:
        logger.debug("_dt_reasoning_block failed (non-fatal): %s", _e)
        return ""


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "feature_engineering", **entry})


def _corr_with_target(feat: pd.Series, y: pd.Series) -> float:
    # Encode string/categorical target to numeric so Pearson corr works.
    # Churn='Yes'/'No', Survived=0/1, etc. — only the numeric case passes through unchanged.
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0].astype(float), index=y.index)
    sub = pd.DataFrame({"f": feat, "y": y}).dropna()
    if len(sub) < 3:
        return 0.0
    if sub["f"].nunique() <= 1 or sub["y"].nunique() <= 1:
        return 0.0
    return float(sub["f"].corr(sub["y"]))


def engineer_features(
    df: pd.DataFrame,
    session: SessionState,
    df_test: pd.DataFrame | None = None,
    raw_train: pd.DataFrame | None = None,
    raw_test: pd.DataFrame | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 5: LLM proposes features, code runs leakage check before each creation.

    Proposal types: interaction, ratio, datetime, domain_combination
    Scenarios:
    - dataset < 200 rows → skip entirely, no LLM call
    - proposals reference dropped columns → silently discard + log
    - all proposals fail leakage check → log, add note to eda_narrative

    Every accept/reject decision and every NaN-fill value is computed from `df`
    (train) only. When `df_test` is given, the identical accepted formulas and
    fill values — never recomputed from test — are applied to it too, and
    `(out_train, out_test)` is returned instead of a single frame.
    """
    task = session.task_config or {}
    target = task.get("target_column")
    target_s = str(target) if target else None
    if not target_s or target_s not in df.columns:
        session.feature_engineering_log = []
        return (df, df_test) if df_test is not None else df

    # CV-gated mode (default): needs the original-scale rows, aligned with `df`.
    if (
        config.FE_SELECTION == "cv"
        and raw_train is not None
        and len(raw_train) == len(df)
        and (df_test is None or (raw_test is not None and len(raw_test) == len(df_test)))
    ):
        return engineer_features_cv(df, session, df_test, raw_train, raw_test)

    n_rows = len(df)

    # ── Decision Router gate — 3-tier check before LLM call ──────────────────
    _router_stats = RouterStats()
    _should_fe, _skip_reason = should_engineer_features(df, session, _router_stats)
    if not _should_fe:
        session.feature_engineering_log = [{"status": "skipped", "reason": _skip_reason}]
        _log(session, {"action": "fe_skipped", "reason": _skip_reason, "n_rows": n_rows})
        log_decision(session, step="Phase 3 - Feature Engineering",
                     action="skipped_by_router", n_rows=n_rows, reason=_skip_reason)
        return (df, df_test) if df_test is not None else df

    y = df[target_s]
    task_type = str(task.get("task_type") or "classification").lower()
    numeric = [c for c in df.columns if c != target_s and pd.api.types.is_numeric_dtype(df[c])]
    dt_cols: list[str] = []
    for c in df.columns:
        if c == target_s:
            continue
        try:
            if np.issubdtype(df[c].dtype, np.datetime64):
                dt_cols.append(c)
        except (TypeError, ValueError):
            pass
        if "date" in str(c).lower() and c not in dt_cols:
            dt_cols.append(c)

    # ── A: Decision-tree reasoning block (NeurIPS 2024 — OCTree technique) ─────
    # Grounds the LLM's proposals in actual data splits rather than EDA narrative alone.
    dt_block = _dt_reasoning_block(df, numeric, y, task_type)
    dt_section = f"\n\n{dt_block}" if dt_block else ""

    client = LLMClient()
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    prompt = f"""Phase 3 Step 5 — Feature Engineering. Propose new derived features.{revision_block}{dt_section}

TASK: {task}
EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2000]}
DATASET_ROWS: {n_rows}

NUMERIC_COLUMNS (use ONLY these for interaction/ratio/domain_combination): {numeric[:40]}
DATETIME_LIKE: {dt_cols}

Propose up to 8 candidate features. For each provide:
- name: short snake_case name
- type: "interaction" | "ratio" | "datetime" | "domain_combination"
- left: column name (MUST be from NUMERIC_COLUMNS)
- right: column name (MUST be from NUMERIC_COLUMNS, empty for datetime or domain_combination)
- formula: Python expression using ONLY names from NUMERIC_COLUMNS (e.g. "SibSp + Parch + 1")
- rationale: one sentence explaining predictive value

Types:
- interaction: left * right  (two NUMERIC columns only)
- ratio: left / right  (two NUMERIC columns only, handles division-by-zero)
- datetime: extract year/month/day/hour/dayofweek from left
- domain_combination: arithmetic expression using ONLY NUMERIC_COLUMNS names

CRITICAL: Do NOT reference string/categorical columns. Only use names from NUMERIC_COLUMNS.
Return JSON only:
{{"features": [{{"name": "...", "type": "...", "left": "...", "right": "...", "formula": "...", "rationale": "..."}}]}}"""

    raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON, json_mode=True)
    parsed = extract_json_object(raw) if raw else None
    proposals: list[dict[str, Any]] = []
    if parsed and isinstance(parsed.get("features"), list):
        proposals = [x for x in parsed["features"] if isinstance(x, dict)]

    out = df.copy()
    out_test = df_test.copy() if df_test is not None else None
    log: list[dict[str, Any]] = []
    leakage_count = 0
    low_corr_count = 0

    for p in proposals[:8]:
        name = str(p.get("name") or "derived")
        typ = str(p.get("type") or "ratio")
        left = p.get("left")
        right = p.get("right")
        formula = str(p.get("formula") or "").strip()
        rationale = str(p.get("rationale") or "")

        # Scenario: references a dropped column → silently discard + log
        referenced_cols = [c for c in [left, right] if c]
        missing_refs = [c for c in referenced_cols if c and c not in out.columns]
        if missing_refs:
            entry = {"feature": name, "status": "discarded_missing_column",
                     "missing": missing_refs, "rationale": rationale}
            log.append(entry)
            _log(session, {"action": "discard_feature_missing_cols", "feature": name,
                           "missing": missing_refs})
            continue

        try:
            if typ == "datetime" and left and left in out.columns:
                part_specs = {
                    f"{name}_{part}": {"type": "datetime", "left": left, "part": part}
                    for part in ("year", "month", "day", "hour", "dayofweek")
                }
                candidates = {c: compute_raw_feature(out, sp) for c, sp in part_specs.items()}
                candidates_test = (
                    {c: compute_raw_feature(out_test, sp) for c, sp in part_specs.items()}
                    if out_test is not None and left in out_test.columns else {}
                )
                for c, vals in candidates.items():
                    fill_val = None
                    if vals.isna().any():
                        fill_val = float(vals.median())
                        vals = vals.fillna(fill_val)
                    cc = float(abs(_corr_with_target(vals, y)))
                    if cc > _LEAKAGE_CORR_THRESHOLD:
                        leakage_count += 1
                        log.append({"feature": c, "status": "rejected_leakage",
                                    "corr": cc, "rationale": rationale})
                        _log(session, {"action": "reject_leakage", "feature": c, "corr": cc})
                    elif cc < _MIN_USEFUL_CORR:
                        low_corr_count += 1
                        log.append({"feature": c, "status": "rejected_low_corr",
                                    "corr": cc, "rationale": rationale})
                        _log(session, {"action": "reject_low_corr", "feature": c, "corr": cc})
                    else:
                        out[c] = vals
                        test_fill = fill_val if fill_val is not None else 0.0
                        log.append({"feature": c, "status": "added", "rationale": rationale,
                                    "spec": {**part_specs[c], "name": c,
                                             "fill": fill_val, "test_fill": test_fill}})
                        _log(session, {"action": "add_feature", "feature": c, "rationale": rationale})
                        if out_test is not None:
                            if c in candidates_test:
                                vals_test = candidates_test[c].fillna(test_fill)
                            else:
                                vals_test = test_fill
                            out_test[c] = vals_test
                continue

            feat_test = None
            can_apply_test = out_test is not None

            spec = {"type": typ, "left": left, "right": right, "formula": formula}
            if typ == "domain_combination" and formula:
                # Arithmetic on numeric columns only (AST-whitelisted, see replay.check_formula).
                feat = compute_raw_feature(out, spec).rename(name)
                # Guard: if formula produced non-numeric output (e.g. string Series),
                # skip — attempting median/corr on strings raises ValueError.
                if not pd.api.types.is_numeric_dtype(feat):
                    logger.debug("engineer_features: skip %s — formula returned non-numeric dtype %s", name, feat.dtype)
                    log.append({"feature": name, "status": "skipped_non_numeric_formula"})
                    continue
                if can_apply_test:
                    try:
                        feat_test = compute_raw_feature(out_test, spec).rename(name)
                    except Exception:
                        feat_test = None
            elif (
                typ in ("ratio", "interaction")
                and left in out.columns
                and right in out.columns
                and pd.api.types.is_numeric_dtype(out[left])
                and pd.api.types.is_numeric_dtype(out[right])
            ):
                feat = compute_raw_feature(out, spec)
                if can_apply_test and left in out_test.columns and right in out_test.columns:
                    feat_test = compute_raw_feature(out_test, spec)
            else:
                log.append({"feature": name, "status": "skipped_unknown_type", "type": typ})
                continue

            # Fill NaN created by ratio (zero denominator) or domain_combination formulas
            fill_val = None
            if feat.isna().any():
                fill_val = float(feat.median())
                feat = feat.fillna(fill_val)
            if feat_test is not None and feat_test.isna().any() and fill_val is not None:
                feat_test = feat_test.fillna(fill_val)

            cc = float(abs(_corr_with_target(feat, y)))
            if cc > _LEAKAGE_CORR_THRESHOLD:
                leakage_count += 1
                log.append({"feature": name, "status": "rejected_leakage",
                            "corr": cc, "rationale": rationale})
                _log(session, {"action": "reject_leakage", "feature": name, "corr": cc})
            elif cc < _MIN_USEFUL_CORR:
                low_corr_count += 1
                log.append({"feature": name, "status": "rejected_low_corr",
                            "corr": cc, "rationale": rationale})
                _log(session, {"action": "reject_low_corr", "feature": name, "corr": cc})
            else:
                out[name] = feat
                log.append({"feature": name, "status": "added", "rationale": rationale,
                            "spec": {**spec, "name": name, "fill": fill_val, "test_fill": fill_val}})
                _log(session, {"action": "add_feature", "feature": name, "rationale": rationale})
                if out_test is not None:
                    out_test[name] = feat_test if feat_test is not None else (fill_val if fill_val is not None else 0.0)

        except Exception as e:
            logger.warning("engineer_features: skip %s: %s", name, e)
            log.append({"feature": name, "status": "error", "detail": str(e)})

    # ── Scenario: all proposals failed leakage check ──────────────────────────
    added = [e for e in log if e.get("status") == "added"]
    if proposals and not added and leakage_count > 0:
        note = "Feature engineering did not yield usable features due to high correlation with target."
        session.eda_narrative = (session.eda_narrative or "") + "\n\n" + note
        _log(session, {"action": "fe_all_leakage", "n_proposals": len(proposals)})
        log_decision(session, step="Phase 3 - Feature Engineering",
                     action="all_proposals_failed_leakage", n_proposals=len(proposals))

    session.feature_engineering_log = log
    session.dataset_profile = compute_dataset_profile(
        out, target_column=target_s)
    session.dataset_state_pointer = "after_feature_engineering"
    log_decision(session, step="Phase 3 - Feature Engineering",
                 action="complete", added=len(added), total_proposals=len(proposals))

    if out_test is not None:
        # Defensive: guarantee identical column set/order to train.
        out_test = out_test.reindex(columns=out.columns, fill_value=0.0)
        return out, out_test
    return out


# ═════════════════════════════════════════════════════════════════════════════
# CV-gated feature engineering (FE_SELECTION=cv)
#
# Candidates are computed from the ORIGINAL-scale columns (feature-selection
# output, before imputation/scaling/encoding) and kept only if they improve the
# cross-validated score of a fast LightGBM proxy on the TRAIN rows — measured
# fold-by-fold against the same folds without the feature. Accepted features
# are added greedily (so redundant ones are rejected), and the measured gains
# are fed back to the LLM for a second round of proposals.
# ═════════════════════════════════════════════════════════════════════════════

_CV_MAX_ROWS = 5_000
_CV_MAX_CANDIDATES_PER_ROUND = 16
_DATETIME_PARTS = ("year", "month", "day", "hour", "dayofweek")


def _proxy_model(task_type: str):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        # Regularized on purpose: an overfit proxy turns informative features into
        # overconfident predictions and hides their gain.
        kw = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=0, verbosity=-1, n_jobs=max(1, (os.cpu_count() or 4) - 2))
        return LGBMClassifier(**kw) if task_type == "classification" else LGBMRegressor(**kw)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if task_type == "classification" else HistGradientBoostingRegressor
        return cls(max_iter=150, learning_rate=0.08, random_state=0)


class _CVScorer:
    """Fold-wise proxy scores on fixed folds, so feature sets are compared paired."""

    def __init__(self, x: pd.DataFrame, y: pd.Series, task_type: str):
        from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

        self.task_type = "classification" if task_type == "classification" else "regression"
        idx = np.arange(len(x))
        if len(idx) > _CV_MAX_ROWS:
            idx = np.sort(np.random.default_rng(0).choice(len(idx), size=_CV_MAX_ROWS, replace=False))
        self.idx = idx
        yv = y.iloc[idx]
        min_class = 0
        if self.task_type == "classification":
            codes, uniques = pd.factorize(yv.astype(str), sort=True)
            self.y = codes
            self.labels = np.arange(len(uniques))
            min_class = int(np.bincount(codes).min()) if len(codes) else 0
        else:
            self.y = pd.to_numeric(yv, errors="coerce").to_numpy(dtype=float)
        repeats = 3 if len(idx) < 2_000 else 2
        if self.task_type == "classification" and min_class >= 3:
            cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=repeats, random_state=0)
            self.splits = list(cv.split(np.zeros(len(idx)), self.y))
        else:
            cv = RepeatedKFold(n_splits=3, n_repeats=repeats, random_state=0)
            self.splits = list(cv.split(np.zeros(len(idx))))
        self.base = x.iloc[idx].to_numpy(dtype=float)

    def scores(self, extra: list[np.ndarray]) -> np.ndarray:
        """Per-fold score (higher is better) of base features + extra columns."""
        from sklearn.metrics import r2_score, roc_auc_score

        xm = np.column_stack([self.base] + [e[self.idx] for e in extra]) if extra else self.base
        out = []
        for tr, va in self.splits:
            model = _proxy_model(self.task_type)
            model.fit(xm[tr], self.y[tr])
            if self.task_type == "classification":
                # ROC-AUC (one-vs-rest for multiclass): measures ranking quality, so a
                # useful feature is not masked by probability-calibration noise.
                proba = model.predict_proba(xm[va])
                full = np.zeros((len(va), len(self.labels)))
                full[:, model.classes_] = proba
                try:
                    if len(self.labels) == 2:
                        out.append(roc_auc_score(self.y[va], full[:, 1]))
                    else:
                        out.append(roc_auc_score(self.y[va], full, multi_class="ovr",
                                                 average="macro", labels=self.labels))
                except ValueError:
                    # A class missing from this fold (depends on y only, so every
                    # feature set falls back identically — comparisons stay paired).
                    out.append(float(np.mean(full.argmax(axis=1) == self.y[va])))
            else:
                out.append(r2_score(self.y[va], model.predict(xm[va])))
        return np.asarray(out)


def _significant_gain(diff: np.ndarray, min_gain: float) -> bool:
    """Mean paired gain above the floor AND above one standard error of the folds."""
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return mean > min_gain and mean > se


def _describe(spec: dict[str, Any]) -> str:
    t = spec.get("type")
    if t == "ratio":
        return f"{spec.get('left')} / {spec.get('right')}"
    if t == "interaction":
        return f"{spec.get('left')} * {spec.get('right')}"
    if t == "datetime":
        return f"{spec.get('left')}.{spec.get('part')}"
    if t == "code":
        return f"code: {str(spec.get('rationale') or '')[:70]}"
    return str(spec.get("formula") or "")


def _code_prompt_block(src: pd.DataFrame) -> str:
    """Prompt section enabling option B (LLM-written feature functions)."""
    cats = []
    for c in src.columns:
        if not pd.api.types.is_numeric_dtype(src[c]):
            vals = [str(v)[:15] for v in src[c].dropna().unique()[:4]]
            cats.append(f"{c!r}: {src[c].nunique()} unique, e.g. {vals}")
    example = (
        '{"name": "charges_minus_contract_mean", "type": "code", "rationale": "...", '
        '"code": "def make_feature(train, test):\\n'
        "    m = train.groupby('Contract')['MonthlyCharges'].mean()\\n"
        "    return (train['MonthlyCharges'] - train['Contract'].map(m),\\n"
        "            test['MonthlyCharges'] - test['Contract'].map(m))\"}"
    )
    return f"""
CATEGORICAL_COLUMNS: {cats[:25] or 'none'}

You may ALSO propose type "code": a Python function for features the other types cannot
express — e.g. a numeric column's mean/median/std/count per category, category frequency,
deviation from the group mean, rank within group, binning, string length or word counts.
Rules for "code" (it runs in an isolated sandbox):
- define exactly: def make_feature(train, test) -> (train_values, test_values)
- train/test are DataFrames with the NUMERIC and CATEGORICAL columns above (no target);
  learn statistics from `train` only and map them onto `test` (unseen categories → NaN is fine)
- return a pd.Series (or a DataFrame with up to 3 numeric columns) per frame, same length/order
- pd, np and math are available; no other imports, no files, no network
Example: {example}
"""


def _cv_proposal_prompt(task, session, n_rows, numeric, dt_cols, dt_section, feedback: str, n: int,
                        code_block: str = "") -> str:
    revision_ctx = (session.preprocessing_meta or {}).get("targeted_revision_context", "")
    revision_block = f"\n\nTARGETED_REVISION_INSTRUCTION: {revision_ctx}" if revision_ctx else ""
    return f"""Phase 3 Step 5 — Feature Engineering. Propose new derived features.{revision_block}{dt_section}

TASK: {task}
EDA_NARRATIVE (truncated): {(session.eda_narrative or '')[:2000]}
DATASET_ROWS: {n_rows}

NUMERIC_COLUMNS (original units, before scaling — use ONLY these): {numeric[:40]}
DATETIME_LIKE: {dt_cols}
{code_block}{feedback}
Every proposal is tested: it is kept only if it improves the cross-validated score of a
gradient-boosting model. Prefer features a tree model cannot easily build itself
(ratios, differences, rates, per-unit quantities, domain formulas{', group statistics' if code_block else ''}).

Propose up to {n} candidate features. For each provide:
- name: short snake_case name (must not clash with existing columns)
- type: "interaction" | "ratio" | "datetime" | "domain_combination"{' | "code"' if code_block else ''}
- left / right: column names from NUMERIC_COLUMNS (right empty for datetime/domain_combination)
- formula: arithmetic expression using ONLY names from NUMERIC_COLUMNS (domain_combination only)
- rationale: one sentence explaining predictive value

Return JSON only:
{{"features": [{{"name": "...", "type": "...", "left": "...", "right": "...", "formula": "...", "rationale": "..."}}]}}"""


def _expand_proposal(p: dict[str, Any], existing: set[str]) -> list[dict[str, Any]]:
    """One proposal → candidate specs (a datetime proposal yields one per date part)."""
    name = str(p.get("name") or "derived").strip()
    typ = str(p.get("type") or "ratio")
    base = {"type": typ, "left": p.get("left"), "right": p.get("right"),
            "formula": str(p.get("formula") or "").strip(), "rationale": str(p.get("rationale") or "")}
    if typ == "datetime":
        specs = [{**base, "name": f"{name}_{part}", "part": part} for part in _DATETIME_PARTS]
    else:
        specs = [{**base, "name": name}]
    return [s for s in specs if s["name"] not in existing]


def _candidate_values(src: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.Series, float | None]:
    """Compute a candidate on the train source; fill NaN/inf with the train median.
    Code features arrive precomputed from the sandbox (spec["_train"])."""
    raw_values = spec["_train"] if spec["type"] == "code" else compute_raw_feature(src, spec)
    feat = finish_feature(raw_values, None).reset_index(drop=True)
    fill_val = float(feat.median()) if feat.isna().any() else None
    if fill_val is not None and np.isnan(fill_val):
        fill_val = 0.0
    return finish_feature(feat, fill_val), fill_val


def _code_candidates(
    proposals: list[dict[str, Any]],
    existing: set[str],
    runner: Any,
    log: list[dict[str, Any]],
    rnd: int,
) -> list[dict[str, Any]]:
    """Statically check the round's "code" proposals, run the safe ones in the sandbox,
    and return one candidate spec per returned column."""
    code_props = [p for p in proposals if str(p.get("type")) == "code"]
    if not code_props:
        return []
    if runner is None:
        for p in code_props:
            log.append({"feature": str(p.get("name")), "status": "skipped_code_disabled", "round": rnd})
        return []
    runnable = []
    for p in code_props:
        name = str(p.get("name") or "code_feature").strip()
        code = normalize_feature_code(str(p.get("code") or ""))
        problem = "name clashes with an existing column" if name in existing else check_feature_code(code)
        if problem:
            # Policy violations are blocked; ordinary mistakes are errors fed back for repair.
            status = "rejected_unsafe_code" if is_safety_problem(problem) else "error"
            log.append({"feature": name, "status": status, "detail": problem, "round": rnd})
            continue
        runnable.append({"name": name, "code": code, "rationale": str(p.get("rationale") or "")})
    if not runnable:
        return []
    results, errors = runner.run(runnable)
    for name, tb in errors.items():
        log.append({"feature": name, "status": "error", "detail": tb, "round": rnd})
    return [
        {"type": "code", "name": c["name"], "code": c["code"], "column": c["column"],
         "input_columns": c["input_columns"], "rationale": c["rationale"],
         "_train": c["train_values"], "_test": c["test_values"]}
        for c in results if c["name"] not in existing
    ]


def engineer_features_cv(
    df: pd.DataFrame,
    session: SessionState,
    df_test: pd.DataFrame | None,
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame | None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """CV-gated feature engineering. `df`/`df_test` are the deep-cleaned frames the
    features are appended to; `raw_train`/`raw_test` are the same rows before deep
    cleaning (feature-selection output), which the features are computed from."""
    task = session.task_config or {}
    target_s = str(task.get("target_column") or "")
    task_type = str(task.get("task_type") or "classification").lower()
    y = df[target_s]
    n_rows = len(df)

    _router_stats = RouterStats()
    should, reason = should_engineer_features(raw_train, session, _router_stats, skip_dense_rule=True)
    if not should:
        session.feature_engineering_log = [{"status": "skipped", "reason": reason}]
        log_decision(session, step="Phase 3 - Feature Engineering",
                     action="skipped_by_router", n_rows=n_rows, reason=reason)
        return (df, df_test) if df_test is not None else df

    src = raw_train.drop(columns=[target_s], errors="ignore").reset_index(drop=True)
    src_test = (raw_test.drop(columns=[target_s], errors="ignore").reset_index(drop=True)
                if raw_test is not None else None)
    numeric = [c for c in src.columns if pd.api.types.is_numeric_dtype(src[c])]
    dt_cols = [c for c in src.columns if "date" in str(c).lower()]
    dt_block = _dt_reasoning_block(raw_train, numeric, raw_train[target_s], task_type)
    dt_section = f"\n\n{dt_block}" if dt_block else ""

    out = df.copy()
    out_test = df_test.copy() if df_test is not None else None
    scorer = _CVScorer(safe_numeric_features(out.drop(columns=[target_s])), y, task_type)
    current = scorer.scores([])
    baseline_mean = float(current.mean())
    accepted_cols: list[np.ndarray] = []

    log: list[dict[str, Any]] = []
    feedback = ""
    client = LLMClient()
    t0 = time.perf_counter()

    # Option B: LLM-written feature functions, executed only in the Docker sandbox.
    code_runner = None
    if code_features_enabled():
        try:
            code_runner = CodeFeatureRunner(src, src_test)
        except Exception as exc:
            logger.warning("Code features disabled — sandbox could not start: %s", exc)
    code_block = _code_prompt_block(src) if code_runner else ""

    try:
        for rnd in range(1, max(1, config.FE_ROUNDS) + 1):
            prompt = _cv_proposal_prompt(task, session, n_rows, numeric, dt_cols, dt_section,
                                         feedback, 8 if rnd == 1 else 6, code_block)
            max_tokens = config.LLM_MAX_TOKENS_CODE if code_runner else config.LLM_MAX_TOKENS_JSON
            raw = client.call(prompt, max_tokens=max_tokens, json_mode=True)
            parsed = extract_json_object(raw) if raw else None
            proposals = [x for x in (parsed or {}).get("features") or [] if isinstance(x, dict)]
            existing = set(out.columns) | set(src.columns)
            candidates = [s for p in proposals if str(p.get("type")) != "code"
                          for s in _expand_proposal(p, existing)]
            candidates += _code_candidates(proposals, existing, code_runner, log, rnd)
            candidates = candidates[:_CV_MAX_CANDIDATES_PER_ROUND]

            # 1. compute each candidate and measure its solo gain over the current set
            scored: list[tuple[float, dict[str, Any], pd.Series, float | None]] = []
            for spec in candidates:
                name = spec["name"]
                try:
                    feat, fill_val = _candidate_values(src, spec)
                    if feat.nunique(dropna=True) <= 1:
                        log.append({"feature": name, "status": "skipped_constant", "round": rnd})
                        continue
                    cc = float(abs(_corr_with_target(feat, y)))
                    if cc > _LEAKAGE_CORR_THRESHOLD:
                        log.append({"feature": name, "status": "rejected_leakage", "corr": cc, "round": rnd})
                        _log(session, {"action": "reject_leakage", "feature": name, "corr": cc})
                        continue
                    diff = scorer.scores(accepted_cols + [feat.to_numpy(dtype=float)]) - current
                    if _significant_gain(diff, config.FE_MIN_CV_GAIN):
                        scored.append((float(diff.mean()), spec, feat, fill_val))
                    else:
                        log.append({"feature": name, "status": "rejected_no_cv_gain", "round": rnd,
                                    "cv_gain": round(float(diff.mean()), 6), "rationale": spec["rationale"],
                                    "describe": _describe(spec)})
                except Exception as e:  # bad column reference, disallowed formula, ...
                    log.append({"feature": name, "status": "error", "detail": str(e)[:200], "round": rnd})

            # 2. greedy: add in order of solo gain, re-checking against the growing set
            for _solo, spec, feat, fill_val in sorted(scored, key=lambda t: -t[0]):
                name = spec["name"]
                trial = scorer.scores(accepted_cols + [feat.to_numpy(dtype=float)])
                diff = trial - current
                if not _significant_gain(diff, config.FE_MIN_CV_GAIN):
                    log.append({"feature": name, "status": "rejected_redundant", "round": rnd,
                                "cv_gain": round(float(diff.mean()), 6), "rationale": spec["rationale"],
                                "describe": _describe(spec)})
                    continue
                accepted_cols.append(feat.to_numpy(dtype=float))
                current = trial
                if spec["type"] == "datetime":
                    test_fill = fill_val if fill_val is not None else 0.0
                else:
                    test_fill = fill_val
                if spec["type"] == "code":
                    # Values come from the sandbox run — LLM code never executes here.
                    full_spec = {k: spec[k] for k in ("type", "code", "column", "input_columns")}
                    full_spec.update({"name": name, "fill": fill_val, "test_fill": test_fill, "source": "raw"})
                    src[name] = feat.to_numpy()
                    if src_test is not None:
                        test_vals = spec["_test"] if spec["_test"] is not None else np.full(len(src_test), np.nan)
                        src_test[name] = finish_feature(test_vals, test_fill).to_numpy()
                else:
                    full_spec = {k: spec[k] for k in ("type", "left", "right", "formula", "part") if spec.get(k)}
                    full_spec.update({"name": name, "fill": fill_val, "test_fill": test_fill, "source": "raw"})
                    # Same function the generated script uses (replay.add_raw_features).
                    src, src_test = add_raw_features(src, src_test, [full_spec])
                out[name] = src[name].to_numpy()
                if out_test is not None and src_test is not None:
                    out_test[name] = np.asarray(src_test[name], dtype=float)
                log.append({"feature": name, "status": "added", "round": rnd,
                            "cv_gain": round(float(diff.mean()), 6), "rationale": spec["rationale"],
                            "describe": _describe(spec), "spec": full_spec})
                _log(session, {"action": "add_feature", "feature": name, "cv_gain": float(diff.mean())})

            round_lines = [
                f"{'ACCEPTED' if e['status'] == 'added' else 'rejected'} {e['feature']} "
                f"[{e.get('describe', '')}]: {e['cv_gain']:+.4f}"
                for e in log if e.get("round") == rnd and "cv_gain" in e
            ]
            round_lines += [
                f"ERROR {e['feature']}: {str(e.get('detail', '')).strip().splitlines()[-1][:160]}"
                for e in log if e.get("round") == rnd and e.get("status") == "error" and e.get("detail")
            ]
            if round_lines:
                feedback = (
                    "\nPREVIOUS ROUND RESULTS (cross-validated gain when added; propose NEW, different "
                    "ideas that build on what was accepted; fix any errors):\n"
                    + "\n".join(round_lines[:20]) + "\n"
                )
    finally:
        if code_runner is not None:
            code_runner.close()

    added = [e for e in log if e.get("status") == "added"]
    session.feature_engineering_log = log
    session.preprocessing_meta["fe_cv_summary"] = {
        "baseline_cv": round(baseline_mean, 6),
        "final_cv": round(float(current.mean()), 6),
        "added": [e["feature"] for e in added],
        "evaluated": len([e for e in log if "cv_gain" in e]),
        "seconds": round(time.perf_counter() - t0, 1),
        "proxy_metric": "roc_auc" if scorer.task_type == "classification" else "r2",
    }
    session.dataset_profile = compute_dataset_profile(out, target_column=target_s)
    session.dataset_state_pointer = "after_feature_engineering"
    log_decision(session, step="Phase 3 - Feature Engineering", action="complete",
                 rationale=(f"CV-gated: {len(added)} feature(s) added; proxy CV "
                            f"{baseline_mean:.4f} -> {float(current.mean()):.4f}"),
                 added=len(added), added_features=[e["feature"] for e in added])

    if out_test is not None:
        out_test = out_test.reindex(columns=out.columns, fill_value=0.0)
        return out, out_test
    return out


