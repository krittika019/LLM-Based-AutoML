"""
Tiered Decision Router
======================
Intercepts preprocessing decisions before they reach the LLM and classifies them:

  Tier 1 – rule :  Completely deterministic from data type / cardinality alone.
  Tier 2 – stat :  Statistical threshold (null %, skew, correlation) is conclusive.
  Tier 3 – llm  :  Genuinely ambiguous — forward to language model.

Concrete columns/decisions handled without LLM (tier 1 or 2):
  • Scaling:      numeric → standard,  non-numeric → none             (always tier-1)
  • Encoding:     numeric → none,  nunique ≤ 2 → one_hot,            (tier-1)
                  nunique ≤ 10 → one_hot,  nunique > 100 → ordinal
  • Imputation:   null_pct == 0 → median/mode,  null_pct < 10 → same (tier-1)
                  null_pct > 70 → flag_as_missing                     (tier-1)
                  10 ≤ null_pct ≤ 30 → median/mode (stat)             (tier-2)
  • Outlier:      non-numeric → leave,  |skew| < 0.75 → leave         (tier-1/2)
                  |skew| ≥ 3.0 → log                                   (tier-2)
  • Feature sel.: zero-variance → drop,  id-like → drop               (tier-1)
                  correlation ≥ 0.97 → drop lower-variance             (tier-2)
  • Feature eng.: < 2 numeric cols, n_rows < 200, plan says no → skip (tier-1)
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constant thresholds ────────────────────────────────────────────────────────
_NULL_LOW_PCT = 10.0       # below: always impute with simple rule
_NULL_MED_PCT = 30.0       # below: stat-tier imputation
_NULL_HIGH_PCT = 70.0      # above: flag_as_missing (no reliable imputation)
_SKEW_MILD = 0.75          # below: leave outliers alone
_SKEW_HEAVY = 3.0          # above: recommend log transform
_ENC_ONE_HOT_MAX = 10      # nunique ≤ this → one_hot (tier-1)
_ENC_ORDINAL_MIN = 100     # nunique > this → ordinal (tier-1)
_CORR_RULE = 0.97          # correlation ≥ this → tier-2 drop
_MIN_ROWS_FE = 200         # fewer rows → skip feature engineering
_MIN_NUMERIC_FE = 2        # fewer numeric cols → skip feature engineering


# ── Column-level route output ──────────────────────────────────────────────────

class ColumnRoute(NamedTuple):
    """
    Preprocessing decision for one feature column.
    Any field that is None was not resolved by the router and must go to the LLM.
    `tier` reflects the *worst* (highest) tier among the resolved fields.
    """
    imputation: str | None   # "median"|"mode"|"knn"|"flag_as_missing"|None
    outlier: str | None      # "iqr_cap"|"log"|"leave"|None
    encoding: str | None     # "one_hot"|"ordinal"|"target"|"none"|None
    scaling: str             # always resolved; "standard"|"minmax"|"none"
    tier: str                # "rule"|"stat"|"llm"
    rationale: str           # human-readable explanation for logging


# ── Router statistics tracker ──────────────────────────────────────────────────

class RouterStats:
    """Accumulates tier counts across a pipeline run for reporting."""

    def __init__(self) -> None:
        self.rule_count = 0
        self.stat_count = 0
        self.llm_count = 0

    def record(self, tier: str) -> None:
        if tier == "rule":
            self.rule_count += 1
        elif tier == "stat":
            self.stat_count += 1
        else:
            self.llm_count += 1

    @property
    def total(self) -> int:
        return self.rule_count + self.stat_count + self.llm_count

    def summary(self) -> dict[str, Any]:
        t = max(self.total, 1)
        return {
            "rule_decisions": self.rule_count,
            "stat_decisions": self.stat_count,
            "llm_decisions": self.llm_count,
            "llm_pct": round(100.0 * self.llm_count / t, 1),
            "avoided_pct": round(100.0 * (self.rule_count + self.stat_count) / t, 1),
        }

    def as_message(self) -> str:
        s = self.summary()
        return (
            f"**Decision Router:** {s['rule_decisions']} rule-based + "
            f"{s['stat_decisions']} stat-based + "
            f"{s['llm_decisions']} LLM decisions — "
            f"**{s['avoided_pct']:.0f}% handled without LLM** "
            f"({s['llm_pct']:.0f}% forwarded to LLM)"
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_identifier_name(col: str) -> bool:
    """Heuristic: column name strongly suggests it is a row identifier."""
    low = col.lower().strip()
    id_suffixes = ("_id", "_key", "_code", "_index", "_uuid", "_guid")
    id_prefixes = ("id_", "key_", "idx_", "row_")
    return (
        low in ("id", "key", "index", "uuid", "guid", "rowid", "row_id")
        or any(low.endswith(s) for s in id_suffixes)
        or any(low.startswith(s) for s in id_prefixes)
        or low.replace("_", "").replace("-", "") in ("id", "key", "index")
    )


def _route_imputation(is_numeric: bool, null_pct: float) -> tuple[str | None, str]:
    """Return (decision, tier_name). None means 'needs LLM'."""
    if null_pct == 0.0:
        decision = "median" if is_numeric else "mode"
        return decision, "rule"
    if null_pct < _NULL_LOW_PCT:
        decision = "median" if is_numeric else "mode"
        return decision, "rule"
    if null_pct > _NULL_HIGH_PCT:
        return "flag_as_missing", "rule"
    if null_pct <= _NULL_MED_PCT:
        # 10–30 %: deterministic enough for median/mode
        decision = "median" if is_numeric else "mode"
        return decision, "stat"
    # 30–70 %: ambiguous → LLM
    return None, "llm"


def _route_outlier(
    is_numeric: bool, skew: float | None, col: str
) -> tuple[str | None, str]:
    """Return (decision, tier_name). None means 'needs LLM'."""
    if not is_numeric:
        return "leave", "rule"
    if _is_identifier_name(col):
        return "leave", "rule"
    if skew is None:
        return None, "llm"
    abs_skew = abs(skew)
    if abs_skew < _SKEW_MILD:
        return "leave", "stat"
    if abs_skew >= _SKEW_HEAVY:
        return "log", "stat"
    return None, "llm"


def _route_encoding(is_numeric: bool, nunique: int) -> tuple[str | None, str]:
    """Return (decision, tier_name). None means 'needs LLM'."""
    if is_numeric:
        return "none", "rule"
    if nunique <= 2:
        return "one_hot", "rule"
    if nunique <= _ENC_ONE_HOT_MAX:
        return "one_hot", "rule"
    if nunique > _ENC_ORDINAL_MIN:
        return "ordinal", "rule"
    # 11–100 unique values: target encoding vs ordinal depends on task type → LLM
    return None, "llm"


def _route_scaling(is_numeric: bool) -> tuple[str, str]:
    """Scaling is always deterministic."""
    return ("standard" if is_numeric else "none"), "rule"


def _worst_tier(*tiers: str) -> str:
    order = {"rule": 0, "stat": 1, "llm": 2}
    return max(tiers, key=lambda t: order.get(t, 2))


# ── Public API ─────────────────────────────────────────────────────────────────

def route_column(
    col: str,
    is_numeric: bool,
    null_pct: float,
    nunique: int,
    skew: float | None,
) -> ColumnRoute:
    """
    Route one column's preprocessing decisions through the tier hierarchy.

    Returns a ColumnRoute where None fields must still go to the LLM.
    Fully resolved columns (no None) never need an LLM call.
    """
    imp, imp_tier = _route_imputation(is_numeric, null_pct)
    out, out_tier = _route_outlier(is_numeric, skew, col)
    enc, enc_tier = _route_encoding(is_numeric, nunique)
    scl, scl_tier = _route_scaling(is_numeric)

    worst = _worst_tier(imp_tier, out_tier, enc_tier, scl_tier)

    reasons: list[str] = []
    if imp_tier != "llm" and imp:
        reasons.append(f"imputation={imp}({imp_tier})")
    if out_tier != "llm" and out:
        reasons.append(f"outlier={out}({out_tier})")
    if enc_tier != "llm" and enc:
        reasons.append(f"encoding={enc}({enc_tier})")
    reasons.append(f"scaling={scl}({scl_tier})")
    rationale = ";  ".join(reasons) or "all fields need LLM"

    return ColumnRoute(
        imputation=imp,
        outlier=out,
        encoding=enc,
        scaling=scl,
        tier=worst,
        rationale=rationale,
    )


def route_column_from_series(col: str, s: pd.Series) -> ColumnRoute:
    """Convenience: compute stats from a live Series then route."""
    is_numeric = bool(pd.api.types.is_numeric_dtype(s))
    null_pct = float(s.isna().mean() * 100)
    nunique = int(s.nunique(dropna=True))
    skew: float | None = None
    if is_numeric:
        clean = s.dropna()
        if len(clean) > 2:
            try:
                skew = float(clean.skew())
            except Exception:
                pass
    return route_column(col, is_numeric, null_pct, nunique, skew)


def route_columns_batch(
    df: pd.DataFrame,
    target: str | None,
    stats: RouterStats,
) -> dict[str, ColumnRoute]:
    """
    Route all non-target columns in *df* and update *stats*.

    Returns {col_name: ColumnRoute}. Columns with any None field need LLM resolution.
    """
    routes: dict[str, ColumnRoute] = {}
    for col in df.columns:
        if col == target:
            continue
        r = route_column_from_series(col, df[col])
        routes[col] = r
        stats.record(r.tier)
        logger.debug("Router [%s] → %s (%s): %s", col, r.tier, "LLM?" if r.tier == "llm" else "RULE/STAT", r.rationale)
    return routes


def needs_llm(route: ColumnRoute) -> bool:
    """True when the route has at least one unresolved field."""
    return route.tier == "llm" or any(
        f is None for f in (route.imputation, route.outlier, route.encoding)
    )


def columns_needing_llm(routes: dict[str, ColumnRoute]) -> list[str]:
    """Return column names that still require an LLM judgment."""
    return [col for col, r in routes.items() if needs_llm(r)]


def build_partial_assignment(
    route: ColumnRoute,
) -> dict[str, str]:
    """
    Convert a ColumnRoute to the assignment dict format used by deep_cleaning.
    Unresolved fields (None) are replaced with cheap defaults that will be
    overwritten by the LLM merge step.
    """
    return {
        "imputation": route.imputation or "median",
        "outlier": route.outlier or "leave",
        "encoding": route.encoding or "none",
        "scaling": route.scaling,
    }


# ── Feature selection routing ──────────────────────────────────────────────────

def route_feature_selection(
    candidates: dict[str, Any],
    stats: RouterStats,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    Split feature-selection candidates into:
      certain_drops  — handled by rule/stat, no LLM needed
      ambiguous_pairs — high-correlation pairs below CORR_RULE threshold

    Returns (certain_drops, ambiguous_pairs).
    `certain_drops` items have keys: column, rationale, tier.
    """
    certain: list[dict[str, str]] = []
    ambiguous: list[dict[str, Any]] = []
    seen: set[str] = set()

    for col in candidates.get("zero_variance") or []:
        if col in seen:
            continue
        seen.add(col)
        certain.append({
            "column": col,
            "rationale": "Near-zero variance — constant feature provides no signal.",
            "tier": "rule",
        })
        stats.record("rule")

    for col in candidates.get("identifier_like") or []:
        if col in seen:
            continue
        seen.add(col)
        certain.append({
            "column": col,
            "rationale": "Cardinality equals row count — pure identifier with no predictive value.",
            "tier": "rule",
        })
        stats.record("rule")

    for pair in candidates.get("high_correlation_pairs") or []:
        r = float(pair.get("correlation") or 0)
        sd = pair.get("suggested_drop")
        sk = pair.get("suggested_keep")
        if not sd:
            continue
        if r >= _CORR_RULE:
            if sd not in seen:
                seen.add(sd)
                certain.append({
                    "column": sd,
                    "rationale": (
                        f"Correlation {r:.3f} with '{sk}' — near-duplicate feature; "
                        "dropping lower-variance copy."
                    ),
                    "tier": "stat",
                })
                stats.record("stat")
        else:
            # Border case (0.95 ≤ r < 0.97): forward to LLM
            ambiguous.append(pair)
            stats.record("llm")

    return certain, ambiguous


# ── Feature engineering gate ───────────────────────────────────────────────────

def should_engineer_features(
    df: pd.DataFrame,
    session: Any,  # SessionState, typed loosely to avoid circular import
    stats: RouterStats,
    skip_dense_rule: bool = False,
) -> tuple[bool, str]:
    """
    Decide whether LLM-based feature engineering is worth attempting.

    `skip_dense_rule` disables the dense-numeric skip — used by CV-gated FE,
    where a candidate that only adds noise is rejected by measurement anyway.

    Returns (proceed: bool, reason: str).
    """
    task = getattr(session, "task_config", {}) or {}
    target = task.get("target_column")
    plan = getattr(session, "preprocessing_plan", {}) or {}

    numeric_cols = [
        c for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    n_rows = len(df)

    # Rule: plan explicitly says no FE
    if plan.get("feature_engineering_recommended") is False:
        stats.record("rule")
        return False, "Preprocessing plan flags feature_engineering_likely=False — skipping LLM call."

    # Rule: dense small all-numeric datasets (e.g. chemistry/sensor measurements).
    # LLM-proposed interactions on already-complete numeric feature sets add noise
    # rather than signal — skip FE to protect raw measurement fidelity.
    non_target_cols = [c for c in df.columns if c != target]
    all_numeric_frac = len(numeric_cols) / max(len(non_target_cols), 1)
    if not skip_dense_rule and all_numeric_frac >= 0.9 and n_rows < 10000 and len(non_target_cols) <= 15:
        stats.record("rule")
        return False, (
            f"Dense all-numeric dataset ({len(non_target_cols)} features, {n_rows} rows) — "
            "skipping feature engineering to avoid adding noise on measurement data."
        )

    # Rule: too few numeric columns to form interactions or ratios
    if len(numeric_cols) < _MIN_NUMERIC_FE:
        stats.record("rule")
        return False, f"Only {len(numeric_cols)} numeric column(s) — not enough for interactions/ratios."

    # Rule: dataset is too small for FE to be reliable
    if n_rows < _MIN_ROWS_FE:
        stats.record("rule")
        return False, f"Dataset has only {n_rows} rows — feature engineering risks overfitting on small datasets."

    # Rule: all remaining features are binary (no meaningful ratios/interactions)
    non_target = [c for c in df.columns if c != target]
    all_binary = all(
        df[c].nunique(dropna=True) <= 2
        for c in non_target
        if pd.api.types.is_numeric_dtype(df[c])
    )
    if all_binary and len(non_target) > 0:
        stats.record("rule")
        return False, "All numeric features are binary — no useful interactions or ratios to engineer."

    # Has datetime-like columns or enough numerics → let LLM decide
    stats.record("llm")
    return True, f"Sufficient numeric features ({len(numeric_cols)}) and rows ({n_rows}) — forwarding to LLM."
