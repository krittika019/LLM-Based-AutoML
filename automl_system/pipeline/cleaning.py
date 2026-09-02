"""Phase 3: basic cleaning — mechanical fixes and ambiguous-case LLM review."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.memory import episodic
from automl_system.state.session import SessionState
from automl_system.utils.llm_json import extract_json_object

logger = logging.getLogger(__name__)


def _log(session: SessionState, entry: dict[str, Any]) -> None:
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, {"phase": "basic_cleaning", **entry})


def _flag_structural_invalids(df: pd.DataFrame, n_rows: int) -> list[dict[str, Any]]:
    """
    Mechanical structural flags:
    - Negative ages
    - Percentages above 100
    - Dates in the future (for historical datasets)
    - Statistically suspicious numeric values (age > 120, etc.)

    Each flag has: column, kind, count, mechanical (True = no LLM needed),
    and pct_of_rows (for 5% ambiguity threshold check).
    """
    issues: list[dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    for col in df.columns:
        name_l = str(col).lower()
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            # Negative ages
            if "age" in name_l:
                bad_neg = (s < 0) & s.notna()
                if bad_neg.any():
                    cnt = int(bad_neg.sum())
                    issues.append({
                        "column": col, "kind": "negative_age", "count": cnt,
                        "mechanical": True, "pct_of_rows": cnt / n_rows * 100,
                    })
                # Statistically suspicious: age > 120
                bad_old = (s > 120) & s.notna()
                if bad_old.any():
                    cnt = int(bad_old.sum())
                    issues.append({
                        "column": col, "kind": "suspicious_age_over_120",
                        "count": cnt, "mechanical": False,
                        "pct_of_rows": cnt / n_rows * 100,
                        "note": "age values > 120 are biologically implausible",
                    })

            # Percentages above 100
            if any(x in name_l for x in ("percent", "pct", "%", "ratio")):
                bad_pct = (s > 100) & s.notna()
                if bad_pct.any():
                    cnt = int(bad_pct.sum())
                    issues.append({
                        "column": col, "kind": "percent_over_100", "count": cnt,
                        "mechanical": True, "pct_of_rows": cnt / n_rows * 100,
                    })

        # Future-date detection for historical datasets
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                future = s.notna() & (s.dt.tz_localize("UTC") > now_utc
                                       if s.dt.tz is None else s > now_utc)
                cnt = int(future.sum())
                if cnt:
                    issues.append({
                        "column": col, "kind": "future_date", "count": cnt,
                        "mechanical": False, "pct_of_rows": cnt / n_rows * 100,
                        "note": "dates after today in a historical dataset",
                    })
            except Exception:
                pass

    return issues


def _coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Try numeric coercion for object columns that look numeric."""
    logs: list[str] = []
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            sample = out[col].dropna().head(50)
            if len(sample) == 0:
                continue
            try:
                conv = pd.to_numeric(out[col], errors="coerce")
                if conv.notna().sum() >= max(1, len(out) * 0.5):
                    out[col] = conv
                    logs.append(f"Coerced column {col!r} to numeric.")
            except Exception:
                pass
    return out, logs


def _resolve_ambiguous_with_llm(
    session: SessionState, ambiguous: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """
    Ask LLM to decide each ambiguous case.
    Returns (summary_string, decisions_list) so the caller can apply the decisions.
    """
    if not ambiguous:
        return "no_ambiguous_cases", []

    client = LLMClient()
    prompt = f"""You are reviewing ambiguous data-quality findings in a dataset.
For each case decide: KEEP, DROP_ROWS, or SET_NULL. Give a one-sentence reason.

Return a JSON array only:
[{{"column": "...", "kind": "...", "decision": "KEEP|DROP_ROWS|SET_NULL", "reasoning": "one sentence"}}]

FINDINGS: {ambiguous[:20]}"""

    raw = client.call(prompt, max_tokens=config.LLM_MAX_TOKENS_JSON, json_mode=True)
    decisions: list[dict[str, Any]] = []
    if raw:
        try:
            parsed = extract_json_object(raw)
            if isinstance(parsed, list):
                decisions = parsed
        except Exception:
            pass

    # Log each decision individually — matched by (column, kind), not position,
    # since the LLM's returned array can omit/reorder entries relative to the
    # findings it was given.
    for amb in ambiguous:
        dec_entry = _match_decision(amb, decisions)
        _log(session, {
            "action": "ambiguous_cleaning_decision",
            "ambiguity": amb.get("kind"),
            "column": amb.get("column"),
            "count": amb.get("count"),
            "pct_of_rows": amb.get("pct_of_rows"),
            "decision": dec_entry.get("decision", "KEEP"),
            "reasoning": dec_entry.get("reasoning", "(no LLM decision)"),
        })

    # Build readable summary
    if decisions:
        lines = [
            f"{d.get('column')} [{d.get('kind', d.get('decision'))}]: {d.get('decision')} — {d.get('reasoning', '')}"
            for d in decisions if isinstance(d, dict)
        ]
        return "\n".join(lines), decisions

    return raw or "no_response", decisions


def _match_decision(amb: dict[str, Any], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Finds the LLM decision for a given finding by (column, kind) identity,
    not list position — the LLM's returned array can omit, reorder, or add
    entries relative to the findings it was given, so positional indexing can
    silently apply the wrong decision to the wrong column."""
    col, kind = amb.get("column"), amb.get("kind")
    for d in decisions:
        if not isinstance(d, dict):
            continue
        if d.get("column") == col and d.get("kind") == kind:
            return d
    for d in decisions:
        if isinstance(d, dict) and d.get("column") == col:
            return d
    return {}


def _apply_llm_decisions(
    out: pd.DataFrame,
    ambiguous: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> pd.DataFrame:
    """Apply DROP_ROWS / SET_NULL decisions returned by the LLM to the DataFrame."""
    now_utc = datetime.now(timezone.utc)
    drop_mask = pd.Series(False, index=out.index)

    for amb in ambiguous:
        dec_entry = _match_decision(amb, decisions)
        action = str(dec_entry.get("decision", "KEEP")).upper()
        if action == "KEEP":
            continue

        col = amb.get("column")
        kind = amb.get("kind")
        if not col or col not in out.columns:
            continue

        s = out[col]
        if kind == "negative_age":
            mask = (s < 0) & s.notna()
        elif kind == "percent_over_100":
            mask = (s > 100) & s.notna()
        elif kind == "suspicious_age_over_120":
            mask = (s > 120) & s.notna()
        elif kind == "future_date" and pd.api.types.is_datetime64_any_dtype(s):
            try:
                mask = s.notna() & (
                    s.dt.tz_localize("UTC") > now_utc if s.dt.tz is None else s > now_utc
                )
            except Exception:
                mask = pd.Series(False, index=out.index)
        else:
            mask = pd.Series(False, index=out.index)

        if action == "DROP_ROWS":
            drop_mask = drop_mask | mask
        elif action == "SET_NULL":
            out = out.copy()
            out.loc[mask, col] = np.nan

    if drop_mask.any():
        out = out[~drop_mask].reset_index(drop=True)

    return out


def basic_clean(df: pd.DataFrame, session: SessionState) -> pd.DataFrame:
    """
    Fix type mismatches, drop exact duplicate rows, flag structural invalids.
    - LLM is skipped entirely when there are no ambiguous cases.
    - Warns if >20% of rows are removed by deduplication.
    - Marks ambiguous when a flag affects >5% of rows.
    Writes session.cleaning_report.
    """
    out = df.copy()
    initial_n = len(out)

    report: dict[str, Any] = {
        "initial_shape": [initial_n, int(out.shape[1])],
        "duplicate_rows_removed": 0,
        "high_dup_warning": False,
        "structural_flags": [],
        "coercion_log": [],
        "ambiguous_cases": [],
        "ambiguous_llm_guidance": "",
    }

    # ── Step 1: Remove exact duplicate rows ──────────────────────────────────
    n_dup = int(out.duplicated().sum())
    if n_dup:
        out = out.drop_duplicates().reset_index(drop=True)
    report["duplicate_rows_removed"] = n_dup

    if n_dup > 0 and initial_n > 0:
        dup_pct = n_dup / initial_n * 100
        if dup_pct > 20.0:
            report["high_dup_warning"] = True
            report["high_dup_pct"] = round(dup_pct, 1)
            report["rows_after_dedup"] = len(out)

    # ── Step 2: Flag structural invalids ─────────────────────────────────────
    n_current = len(out)
    flags = _flag_structural_invalids(out, n_current) if n_current > 0 else []
    report["structural_flags"] = flags

    # ── Step 3: Type coercion for numeric-as-string columns ──────────────────
    out, coercions = _coerce_types(out)
    report["coercion_log"] = coercions

    # ── Step 4: Classify ambiguous vs mechanical flags ────────────────────────
    mechanical_flags: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for flag in flags:
        pct = flag.get("pct_of_rows", 0.0)
        if flag.get("mechanical"):
            if pct > 5.0:
                # Too many rows affected — surface as ambiguous rather than silently fixing
                ambiguous.append({
                    **flag,
                    "reason": f"mechanical fix would affect {pct:.1f}% of rows (>5% threshold)",
                })
            else:
                mechanical_flags.append(flag)
        else:
            ambiguous.append({**flag, "reason": flag.get("note", "non-mechanical finding")})

    report["ambiguous_cases"] = ambiguous

    # ── Step 5: Apply mechanical fixes ───────────────────────────────────────
    for flag in mechanical_flags:
        col = flag["column"]
        kind = flag["kind"]
        if kind == "negative_age" and col in out.columns:
            out.loc[out[col] < 0, col] = np.nan
        elif kind == "percent_over_100" and col in out.columns:
            name_l = col.lower()
            if any(x in name_l for x in ("percent", "pct", "%", "ratio")):
                out.loc[out[col] > 100, col] = np.nan

    _log(session, {
        "action": "mechanical_fixes_applied",
        "count": len(mechanical_flags),
        "fixed": [{"column": f["column"], "kind": f["kind"]} for f in mechanical_flags],
    })

    # ── Step 6: LLM for ambiguous cases (skipped if empty) ───────────────────
    if ambiguous:
        guidance, llm_decisions = _resolve_ambiguous_with_llm(session, ambiguous)
        report["ambiguous_llm_guidance"] = guidance
        # Apply the LLM's DROP_ROWS / SET_NULL decisions to the DataFrame
        rows_before = len(out)
        out = _apply_llm_decisions(out, ambiguous, llm_decisions)
        rows_removed = rows_before - len(out)
        if rows_removed:
            _log(session, {"action": "llm_decisions_applied",
                           "rows_removed": rows_removed})
    else:
        report["ambiguous_llm_guidance"] = "no_ambiguous_cases"
        _log(session, {"action": "llm_skipped", "reason": "no ambiguous cases found"})

    session.cleaning_report = report
    session.expected_shape = (len(out), out.shape[1])
    session.preprocessing_meta["basic_clean_shape"] = list(session.expected_shape)
    _log(session, {
        "action": "basic_clean_complete",
        "final_shape": list(session.expected_shape),
        "duplicate_rows_removed": n_dup,
        "ambiguous_cases_count": len(ambiguous),
    })
    return out
