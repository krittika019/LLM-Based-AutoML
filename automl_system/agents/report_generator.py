"""Phase 7: produce a plain-English markdown report for the full run."""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from automl_system import config
from automl_system.llm.client import LLMClient
from automl_system.state.session import SessionState
from automl_system.utils.decision_log import compact_decisions, merge_logs_for_report
from automl_system.utils.llm_json import extract_json_object
from automl_system.utils.profile_conditioner import condition_profile_for_report

logger = logging.getLogger(__name__)


def _best_result(session: SessionState) -> dict[str, Any] | None:
    # Source from execution_results' own best_model_name/best_test_score, set by
    # execution.py BEFORE it appends AutoGluon/FLAML reference rows to
    # comparison_table. Re-deriving "best" by scanning comparison_table instead
    # would wrongly pick a reference row whenever AutoGluon/FLAML outscores our
    # own candidates — comparison_table exists to inform the user, not to name
    # our own pipeline's winner.
    results = session.execution_results or {}
    model = results.get("best_model_name")
    score = results.get("best_test_score")
    if model is None or not isinstance(score, (int, float)):
        return None
    return {"model": model, "test_score": score}


# Column names that carry no meaning for a stakeholder. When the resolved target
# is one of these, the raw name is withheld from the LLM entirely — instructing it
# not to print the name proved unreliable, it would still echo back "the 'target'
# column" from the task config it was shown.
_GENERIC_TARGET_NAMES = {
    "target", "label", "labels", "y", "class", "outcome",
    "response", "dependent_variable",
}

# External AutoML systems appended to comparison_table for benchmarking only.
# They are not candidates the pipeline chose, so they are withheld from the
# report — otherwise the LLM names whichever row scored highest as "the winner".
_BENCHMARK_ROW_MARKERS = ("autogluon", "flaml")

# Heading text is owned here, not by the model. Asked to emit its own headings
# the model renamed them, merged two sections, and appended a "Conclusion" that
# only restated the rest — so it now returns one JSON field per section and the
# document is assembled below.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("user_request", "## User Request and Task Understanding"),
    ("task_and_challenges", "## Core Task and Data Challenges"),
    ("eda_and_preprocessing", "## EDA Findings and Preprocessing Decisions Applied"),
    ("feature_engineering", "## Feature Engineering"),
    ("model_selection", "## Model Selection and Evaluation Metric"),
    ("takeaway", "## Takeaway"),
)
_REQUIRED_HEADINGS = tuple(heading for _key, heading in _SECTIONS)


def _parse_eda_narrative(narrative: str) -> dict[str, str]:
    """EDA stores its narrative as a stringified dict. Parse it back so each
    finding can be labelled in the prompt — passing the raw repr made the model
    skim it and hedge ("an unspecified third feature") despite the names being
    present.
    """
    text = (narrative or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return {str(k): str(v).strip() for k, v in parsed.items() if str(v).strip()}
        except (ValueError, SyntaxError):
            pass
    return {"summary": text}


def _eda_top_features(session: SessionState, eda_sections: dict[str, str]) -> list[str]:
    """Feature names EDA called most predictive, matched against real column
    names so the list can never contain an invented feature. Returned in the
    order they appear in the narrative.
    """
    sentence = eda_sections.get("top_features") or ""
    if not sentence:
        return []
    profile = session.initial_dataset_profile or session.dataset_profile or {}
    columns = [c for c in (profile.get("columns") or {}) if c]
    hits = [(sentence.find(c), c) for c in columns if c in sentence]
    return [c for pos, c in sorted(hits) if pos >= 0]


def _mi_dropped_features(decisions: list[dict[str, Any]]) -> list[str]:
    """Features execution dropped for near-zero mutual information, recovered
    from the decision log's rationale string.
    """
    for entry in decisions:
        if str(entry.get("action") or "") != "mi_feature_filter":
            continue
        match = re.search(r"\[(.*?)\]", str(entry.get("rationale") or ""), re.S)
        if not match:
            return []
        try:
            parsed = ast.literal_eval("[" + match.group(1) + "]")
        except (ValueError, SyntaxError):
            return []
        return [str(x) for x in parsed]
    return []


def _eda_execution_conflicts(eda_top: list[str], mi_dropped: list[str]) -> list[str]:
    """Features EDA ranked as top predictors that execution later discarded.

    EDA ranks by correlation on raw columns; the MI filter runs after encoding,
    so a name matches either exactly or as the stem of an encoded column. The
    overlap is the most interesting thing in the run and the report should
    explain it rather than quietly present both facts as if they agree.
    """
    conflicts = []
    for feature in eda_top:
        if any(d == feature or d.startswith(f"{feature}_") for d in mi_dropped):
            conflicts.append(feature)
    return conflicts


def _preprocessing_facts(session: SessionState, decisions: list[dict[str, Any]]) -> str:
    """Concrete record of what preprocessing actually did, grouped by method.

    Built in code because the raw per-column assignment dict is too large and
    repetitive for the model to summarise faithfully — given the dict it either
    described one column or generalised into "standard preprocessing was applied".
    """
    lines: list[str] = []

    dropped = session.feature_selection_log or []
    if dropped:
        for entry in dropped[:8]:
            reason = str(entry.get("rationale") or "").strip() or "low predictive value"
            lines.append(f"- Dropped '{entry.get('column')}' during feature selection: {reason}")
    else:
        lines.append("- Feature selection kept every column; none were dropped.")

    by_method: dict[str, dict[str, list[str]]] = {
        "imputation": {}, "encoding": {}, "scaling": {}, "outlier": {},
    }
    for column, assignment in (session.deep_cleaning_assignments or {}).items():
        if not isinstance(assignment, dict):
            continue
        for dimension in by_method:
            value = str(assignment.get(dimension) or "").strip().lower()
            if value in ("", "none", "leave"):
                continue
            by_method[dimension].setdefault(value, []).append(str(column))

    labels = {
        "imputation": "Filled missing values using",
        "encoding": "Encoded with",
        "scaling": "Scaled using",
        "outlier": "Outliers handled with",
    }
    for dimension, groups in by_method.items():
        for method, columns in groups.items():
            shown = ", ".join(columns[:6]) + (f" (+{len(columns) - 6} more)" if len(columns) > 6 else "")
            lines.append(f"- {labels[dimension]} {method}: {shown}")

    mi_dropped = _mi_dropped_features(decisions)
    if mi_dropped:
        lines.append(
            f"- Dropped {len(mi_dropped)} feature(s) before training for near-zero mutual "
            f"information with the outcome: {', '.join(mi_dropped)}"
        )

    return "\n".join(lines)


def _feature_engineering_facts(session: SessionState, decisions: list[dict[str, Any]]) -> str:
    """What the feature engineering stage produced, or the recorded reason it produced nothing.

    Split out of _preprocessing_facts because this now owns a section of its own.
    The empty case matters more than the populated one: given no recorded reason
    the model filled the gap by speculating about its own pipeline ("the reason
    for skipping this step is not explicitly stated, but it implies that..."), so
    when nothing is on record the payload says so and forbids the guess outright.
    """
    fe_log = session.feature_engineering_log or []
    added = [e for e in fe_log if isinstance(e, dict) and e.get("status") == "added"]

    if added:
        lines = [f"- {len(added)} new feature(s) were created:"]
        for entry in added[:10]:
            name = str(entry.get("column") or entry.get("name") or "?")
            reason = str(entry.get("rationale") or entry.get("note") or "").strip()
            lines.append(f"  - {name}" + (f" — {reason}" if reason else ""))
        if len(added) > 10:
            lines.append(f"  - (+{len(added) - 10} more)")
        return "\n".join(lines)

    # Nothing was created. The reason is usually on the log entry itself — a
    # stage-level {"status": "skipped", "reason": ...} row — so that is read first
    # and the decision log is only a fallback. Reading the decision log alone left
    # the reason out of the payload and the model narrated its absence instead.
    reason = ""
    for entry in fe_log:
        if not isinstance(entry, dict) or entry.get("status") == "added":
            continue
        reason = str(entry.get("reason") or entry.get("rationale") or "").strip()
        if reason:
            break
    if not reason:
        for entry in decisions:
            action = str(entry.get("action") or "").lower()
            phase = str(entry.get("phase") or entry.get("step") or "").lower()
            if "skip" in action and ("feature" in action or "feature_engineering" in phase):
                reason = str(
                    entry.get("reason") or entry.get("rationale") or entry.get("note") or ""
                ).strip()
                if reason:
                    break

    # A stage-level skip row is not a rejected candidate. Only entries that name a
    # column are real proposals; counting the skip row as one produced "1 candidate
    # was proposed and rejected: ?" and invited the model to explain the "?".
    rejected = [
        e for e in fe_log
        if isinstance(e, dict) and e.get("status") != "added" and (e.get("column") or e.get("name"))
    ]
    lines = ["- No new features were created."]
    if reason:
        lines.append(f"- Recorded reason: {reason}")
    if rejected:
        names = ", ".join(str(e.get("column") or e.get("name")) for e in rejected[:6])
        lines.append(f"- {len(rejected)} candidate(s) were proposed and rejected: {names}")
    if not reason and not rejected:
        # Phrased as a fact, not as an instruction: this block is rendered verbatim
        # into the deterministic fallback report, so anything written at the model
        # would be shown to the reader. The rule against guessing a reason lives in
        # the prompt instead.
        lines.append("- No reason for this is on record.")
    return "\n".join(lines)


def _ranked_models(all_results: list[dict[str, Any]]) -> list[tuple[str, float, Any]]:
    """Final-round models ranked by test score, external benchmark rows excluded.

    Shared by the prompt payload and the post-generation check so both agree on
    which three models the report is required to name.
    """
    if not all_results:
        return []
    scored: list[tuple[str, float, Any]] = []
    for row in (all_results[-1].get("comparison_table") or []):
        name = str(row.get("model_name") or "").strip()
        test = row.get("test_score")
        if not name or not isinstance(test, (int, float)):
            continue
        if any(marker in name.lower() for marker in _BENCHMARK_ROW_MARKERS):
            continue
        scored.append((name, float(test), row.get("cv_score")))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _model_scores_block(all_results: list[dict[str, Any]], best: dict[str, Any] | None) -> str:
    """Ranked model scores for the final iteration, excluding external benchmark rows.

    Ranking, runner-up, and the winning gap are computed here rather than left to
    the model: asked to read them off an unordered table it named the wrong
    runner-up and overstated the gap fivefold, which would misrepresent how
    close the result actually was.
    """
    scored = _ranked_models(all_results)
    if not scored:
        return ""
    top = scored[:3]

    # The top three are broken out explicitly because the report section asks for
    # them by name. Left to pick "the best few" out of a full list the model
    # reported two, or three that were not the top three.
    lines = [
        f"- TOP {len(top)} — name every one of these in the report, each with its "
        f"test score, in this order:"
    ]
    for rank, (name, test, cv) in enumerate(top, start=1):
        cv_text = f"{cv:.4f}" if isinstance(cv, (int, float)) else "not recorded"
        lines.append(f"    #{rank} {name}: held-out test {test:.4f}, cross-validation {cv_text}")

    if len(scored) > len(top):
        rest = ", ".join(f"{name} ({test:.4f})" for name, test, _cv in scored[len(top):])
        lines.append(f"- Also trained, all scoring below the top {len(top)}: {rest}")

    winner_name, winner_score, _ = top[0]
    if len(top) > 1:
        runner_name, runner_score, _ = top[1]
        lines.append(
            f"- WINNER: {winner_name} ({winner_score:.4f}). RUNNER-UP: {runner_name} "
            f"({runner_score:.4f}). GAP: {winner_score - runner_score:.4f} — use these "
            f"three facts verbatim; do not re-rank the list yourself."
        )
    if best and best.get("model") and str(best["model"]) != winner_name:
        # The selected model is normally the top scorer; if the pipeline chose
        # otherwise, say so rather than letting the report infer a winner.
        lines.append(f"- SELECTED BY THE PIPELINE: {best['model']} at {best['test_score']:.4f}")
    if len(all_results) > 1:
        lines.append(f"- Scores are from the final of {len(all_results)} execution rounds.")
    return "\n".join(lines)


def _majority_baseline(all_results: list[dict[str, Any]], best: dict[str, Any] | None) -> float | None:
    """Accuracy a majority-class guesser would reach, from the confusion matrix
    row sums. Returned separately so the audit can check the report actually
    stated it.
    """
    grid = _best_confusion_matrix(all_results, best)
    if not grid:
        return None
    total = sum(sum(r) for r in grid)
    return (max(sum(r) for r in grid) / total) if total > 0 else None


def _best_confusion_matrix(
    all_results: list[dict[str, Any]], best: dict[str, Any] | None
) -> list[list[int]] | None:
    """Confusion matrix for the selected model, as ints, or None if unavailable."""
    if not all_results:
        return None
    rows = all_results[-1].get("comparison_table") or []
    target_name = str((best or {}).get("model") or "")
    row = next(
        (r for r in rows if str(r.get("model_name") or "") == target_name),
        next((r for r in rows if r.get("is_best")), None),
    )
    if not isinstance(row, dict):
        return None
    metrics = row.get("detailed_metrics")
    if not isinstance(metrics, dict):
        return None
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return None
    try:
        return [[int(v) for v in r] for r in matrix]
    except (TypeError, ValueError):
        return None


def _measured_facts_line(all_results: list[dict[str, Any]], best: dict[str, Any] | None) -> str:
    """Baseline and error counts as a reader-facing line, built entirely in code.

    Rendered into the finished document rather than requested from the model, for
    the same reason the title and headings are: these are the numbers that make
    the score interpretable, and they were the ones the model most often dropped
    or replaced with a vague quantifier.
    """
    grid = _best_confusion_matrix(all_results, best)
    if not grid:
        return ""
    total = sum(sum(r) for r in grid)
    if total <= 0:
        return ""
    baseline = max(sum(r) for r in grid) / total

    parts = [
        f"**Measured on {total} held-out rows.** A majority-class guesser would "
        f"score {baseline:.4f} accuracy."
    ]
    accuracy = None
    rows = all_results[-1].get("comparison_table") or []
    target_name = str((best or {}).get("model") or "")
    row = next(
        (r for r in rows if str(r.get("model_name") or "") == target_name),
        next((r for r in rows if r.get("is_best")), {}),
    )
    metrics = row.get("detailed_metrics") if isinstance(row, dict) else None
    if isinstance(metrics, dict) and isinstance(metrics.get("accuracy"), (int, float)):
        accuracy = float(metrics["accuracy"])
    if accuracy is not None:
        parts.append(f"This model scores {accuracy:.4f} ({accuracy - baseline:+.4f}).")

    if len(grid) == 2 and len(grid[0]) == 2:
        (tn, fp), (fn, tp) = grid[0], grid[1]
        actual_pos = tp + fn
        share = f" ({tp / actual_pos:.1%})" if actual_pos else ""
        parts.append(
            f"It catches {tp} of {actual_pos}{share} actual positive cases, misses "
            f"{fn}, and raises {fp} false alarms against {tn} correctly cleared."
        )
    return " ".join(parts)


def _error_profile_block(all_results: list[dict[str, Any]], best: dict[str, Any] | None) -> str:
    """Baseline comparison and error breakdown for the selected model.

    Both are derived from the confusion matrix execution already stores in
    detailed_metrics, so neither costs an extra fit: the matrix row sums give the
    per-class support, and the largest of them is exactly what a majority-class
    guesser would score. Without this the report states a score with nothing to
    read it against, which is the single most common defect in an ML report.
    """
    grid = _best_confusion_matrix(all_results, best)
    if not grid:
        return ""
    total = sum(sum(r) for r in grid)
    support = [sum(r) for r in grid]
    if total <= 0 or not support:
        return ""

    rows = all_results[-1].get("comparison_table") or []
    target_name = str((best or {}).get("model") or "")
    row = next(
        (r for r in rows if str(r.get("model_name") or "") == target_name),
        next((r for r in rows if r.get("is_best")), {}),
    )
    metrics = row.get("detailed_metrics") if isinstance(row, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}

    lines = [f"- Held-out test set: {total} rows."]
    baseline = max(support) / total
    # Phrased as facts only: this block is rendered verbatim into the deterministic
    # fallback report, so anything addressed at the model would reach the reader.
    lines.append(
        f"- Majority-class baseline accuracy: {baseline:.4f} — what a model that always "
        f"guesses the most common class would score."
    )
    accuracy = metrics.get("accuracy")
    if isinstance(accuracy, (int, float)):
        lines.append(
            f"- Selected model accuracy: {accuracy:.4f}, which is "
            f"{accuracy - baseline:+.4f} against that baseline."
        )

    if len(grid) == 2 and len(grid[0]) == 2:
        (tn, fp), (fn, tp) = grid[0], grid[1]
        lines.append(
            f"- Error breakdown on the positive class: {tp} caught, {fn} missed "
            f"(false negatives), {fp} false alarms, {tn} correctly cleared."
        )
        actual_pos = tp + fn
        if actual_pos:
            lines.append(
                f"- The model finds {tp}/{actual_pos} of the actual positive cases "
                f"({tp / actual_pos:.1%}). The decision threshold was not tuned."
            )
    else:
        misses = total - sum(grid[i][i] for i in range(min(len(grid), len(grid[0]))))
        lines.append(f"- {misses} of {total} test rows were assigned the wrong class.")
    return "\n".join(lines)


# Keywords that identify which canonical section a model-invented heading means.
# Matching is on the heading's own words only: renaming by position looked
# tempting but silently relabelled every section the one time the model swapped
# "The Data" for a trailing "Conclusion".
_HEADING_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("## User Request and Task Understanding", ("request", "asked", "understanding", "objective")),
    ("## Core Task and Data Challenges",
     ("challenge", "core task", "difficult", "obstacle", "data overview", "data profile")),
    ("## EDA Findings and Preprocessing Decisions Applied",
     ("eda", "exploratory", "preprocess", "cleaning", "prepared")),
    ("## Feature Engineering", ("feature engineering", "engineered", "new feature")),
    ("## Model Selection and Evaluation Metric",
     ("model", "metric", "evaluation", "result", "performance")),
    ("## Takeaway", ("takeaway", "bottom line", "conclusion", "what it means")),
)


def _normalize_headings(body: str) -> str:
    """Rename model-invented headings onto the canonical set where unambiguous.

    Only rewrites a heading when its own wording identifies exactly one
    canonical section and that section has not already been claimed, so content
    is never relabelled as a section it does not contain.
    """
    lines = body.splitlines()
    claimed: set[str] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not re.match(r"^##\s+\S", stripped):
            continue
        text = stripped.lstrip("#").strip().lower()
        matches = [
            canonical for canonical, keywords in _HEADING_KEYWORDS
            if canonical not in claimed and any(word in text for word in keywords)
        ]
        if len(matches) == 1:
            claimed.add(matches[0])
            lines[index] = matches[0]
    return "\n".join(lines)


# Key names the model has substituted for the requested ones. Kept as a fallback
# behind the explicit schema in the prompt: without it a renamed key silently
# drops a whole section from the finished report.
_SECTION_KEY_ALIASES: dict[str, str] = {
    "introduction": "user_request",
    "overview": "user_request",
    "what_was_asked": "user_request",
    "request": "user_request",
    "user_request_and_task_understanding": "user_request",
    "task_understanding": "user_request",
    "goal": "user_request",
    "core_task_and_data_challenges": "task_and_challenges",
    "challenges": "task_and_challenges",
    "data_challenges": "task_and_challenges",
    "core_task": "task_and_challenges",
    "the_data": "task_and_challenges",
    "data": "task_and_challenges",
    "dataset": "task_and_challenges",
    "eda": "eda_and_preprocessing",
    "eda_findings": "eda_and_preprocessing",
    "preprocessing": "eda_and_preprocessing",
    "data_preprocessing": "eda_and_preprocessing",
    "preprocessing_decisions": "eda_and_preprocessing",
    "what_the_system_changed": "eda_and_preprocessing",
    "what_changed": "eda_and_preprocessing",
    "feature_engineering_decisions": "feature_engineering",
    "new_features": "feature_engineering",
    "engineered_features": "feature_engineering",
    "models": "model_selection",
    "results": "model_selection",
    "models_and_results": "model_selection",
    "model_selection_and_evaluation_metric": "model_selection",
    "evaluation": "model_selection",
    "evaluation_metric": "model_selection",
    "conclusion": "takeaway",
    "summary": "takeaway",
    "drivers": "takeaway",
    "key_features": "takeaway",
    "what_drives_the_prediction": "takeaway",
}


def _resolve_section_keys(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map returned keys onto the canonical section keys.

    Values that come back as nested objects rather than strings are flattened to
    their text, since the whole object would otherwise be rendered verbatim into
    the report as a Python dict.
    """
    canonical = {key for key, _heading in _SECTIONS}
    out: dict[str, Any] = {}
    for key, value in parsed.items():
        name = _SECTION_KEY_ALIASES.get(str(key).strip().lower(), str(key).strip().lower())
        if name not in canonical or out.get(name):
            continue
        if isinstance(value, dict):
            value = " ".join(str(v).strip() for v in value.values() if str(v).strip())
        elif isinstance(value, list):
            value = " ".join(str(v).strip() for v in value if str(v).strip())
        out[name] = value
    return out


def _restore_encoded_names(body: str, feature_names: list[str]) -> str:
    """Put the category suffix back on encoded feature names in the takeaway.

    Repairable where the other checks are not: the correct string is known, so
    nothing is invented by writing it. Scoped to the takeaway because the bare
    stem is the real raw column name and is correct wherever the report discusses
    preprocessing — "one-hot encoded ST_Slope" must not become "ST_Slope_Up".

    The model kept writing "ST_Slope" for "ST_Slope_Up" after three separate
    instructions not to, which turns an indicator for one category into a
    quantity that can be higher or lower — a different claim than the importances
    support.
    """
    suffixed = [f for f in feature_names if "_" in f]
    if not suffixed:
        return body
    lines = body.splitlines()
    start = next(
        (i for i, l in enumerate(lines) if l.strip().lower() == "## takeaway"),
        None,
    )
    if start is None:
        return body
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("## ")),
        len(lines),
    )
    segment = "\n".join(lines[start:end])
    for feature in suffixed:
        stem = feature.rsplit("_", 1)[0]
        if len(stem) < 3 or feature in segment:
            continue
        # Negative lookahead so a name already carrying any suffix is left alone.
        segment = re.sub(rf"\b{re.escape(stem)}\b(?!_)", feature, segment)
    return "\n".join(lines[:start] + segment.splitlines() + lines[end:])


def _scrub_generic_target(text: str, name: str) -> str:
    """Last-resort removal of a meaningless internal column name from prose."""
    if not name:
        return text
    escaped = re.escape(name)
    for pattern in (
        rf"""the\s+["']{escaped}["']\s+column""",
        rf"""["']{escaped}["']\s+column""",
        rf"""the\s+{escaped}\s+column""",
    ):
        text = re.sub(pattern, "the outcome being predicted", text, flags=re.IGNORECASE)
    return text


# Phrasing that signals the model filled a section with something other than the
# facts it was given. Each entry was observed in a real run before the rule that
# now forbids it was added, so the list doubles as the checklist the prompt is
# written against.
_HEDGE_PHRASES: tuple[str, ...] = (
    "unspecified", "not specified", "a certain column", "several features",
    "another feature", "and others", "among others", "significant margin",
    "wide margin", "did not significantly impact",
    # Speculation about the pipeline's own reasoning: the report narrating a
    # guess about the run that produced it reads as the system not knowing what
    # it did.
    "not explicitly stated", "not stated", "it implies", "presumably",
)

# Caveats with no number behind them. These read as analysis but carry none, and
# they were the filler the model reached for whenever a section had too little
# real content to fill.
_UNSUPPORTED_CAVEATS: tuple[str, ...] = (
    "may affect", "might affect", "could affect", "may impact",
    "could introduce", "some distributional", "potentially",
    "it is possible that",
)

# Language that actually addresses a stage disagreement rather than just naming
# the feature involved — the feature already appears in the dropped-column list,
# so its presence alone does not show the conflict was explained.
_CONFLICT_EXPLANATION_WORDS: tuple[str, ...] = (
    "redundant", "redundancy", "already", "once the other", "on its own",
    "individually", "overlap", "alongside",
)


def _section_text(body: str, heading: str) -> str:
    """Body text under one heading, up to the next heading. Empty when absent."""
    lines = (body or "").splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if capturing:
                break
            capturing = stripped.lower() == heading.strip().lower()
            continue
        if capturing:
            out.append(line)
    return "\n".join(out)


def audit_report_text(
    body: str,
    *,
    expected_models: list[str] | None = None,
    conflicts: list[str] | None = None,
    expected_features: list[str] | None = None,
    baseline: float | None = None,
) -> dict[str, list[str]]:
    """Check a finished report body against the constraints it was written under.

    Returns check name -> offending items; an empty list means the check passed.
    Detection only — nothing is repaired, because every repair here would have to
    invent the fact the model failed to use.

    Split out of generate_report so a saved report can be re-audited without
    re-running the pipeline, and so violation counts can be compared across
    prompt variants.
    """
    lowered = (body or "").lower()
    findings: dict[str, list[str]] = {
        "missing_sections": [h for h in _REQUIRED_HEADINGS if h.lower() not in lowered],
        "hedges": [p for p in _HEDGE_PHRASES if p in lowered],
        "unsupported_caveats": [p for p in _UNSUPPORTED_CAVEATS if p in lowered],
        "directional_claims": [],
        "unnamed_top_models": [],
        "unexplained_conflict": [],
        "truncated_feature_names": [],
        "missing_baseline": [],
    }

    # "ST_Slope_Up" is an indicator for one category; shortened to "ST_Slope" it
    # reads as a quantity that can be higher or lower, which is a different and
    # unsupported claim. Scoped to the takeaway, because the bare stem is the real
    # raw column name and is correct everywhere the report discusses preprocessing
    # — checking the whole document flagged a correct sentence.
    takeaway_text = _section_text(body, "## Takeaway").lower()
    for feature in (expected_features or []):
        if "_" not in feature:
            continue
        stem = feature.rsplit("_", 1)[0].lower()
        if len(stem) < 3:
            continue
        if feature.lower() not in takeaway_text and re.search(rf"\b{re.escape(stem)}\b", takeaway_text):
            findings["truncated_feature_names"].append(f"{stem} (should be {feature})")

    # A score with nothing to read it against is the most common defect in an ML
    # report, so when a baseline was available its absence is itself a finding.
    # The number is required, not the word: "a significant improvement over the
    # baseline" names the comparison while withholding the only part of it a
    # reader can act on.
    if baseline is not None:
        # Three significant digits minimum. A two-digit form like "55" matches
        # inside any unrelated percentage in the document and passed a report that
        # never stated the baseline at all.
        printed = {f"{baseline:.4f}", f"{baseline:.3f}", f"{baseline * 100:.1f}"}
        if not any(p in lowered for p in printed):
            findings["missing_baseline"] = [f"{baseline:.4f} was available but its value was never stated"]

    # Importance scores carry no direction, so any higher/lower risk claim is
    # invented — worth flagging loudly before a clinical reader sees it.
    match = re.search(r"\b(higher|lower|increas\w+|decreas\w+)\b[^.]{0,60}\brisk\b", lowered)
    if match:
        findings["directional_claims"] = [match.group(0)]

    # The model selection section is required to name all three; naming two, or
    # naming three that did not place, is what the TOP block exists to prevent.
    if expected_models:
        findings["unnamed_top_models"] = [
            name for name in expected_models if name.lower() not in lowered
        ]

    if conflicts:
        explained = any(word in lowered for word in _CONFLICT_EXPLANATION_WORDS)
        named = [c for c in conflicts if c.lower() in lowered]
        if not (named and explained):
            findings["unexplained_conflict"] = list(conflicts)

    return findings


def generate_report(session: SessionState, all_results: list[dict[str, Any]]) -> str:
    """
    Generate a markdown narrative covering data, decisions, model performance, iterations, and recommendation.
    """
    decisions = merge_logs_for_report(session)
    if len(decisions) < 5:
        logger.warning(
            "decisions log is suspiciously short (%d entries) — report may be incomplete",
            len(decisions),
        )

    # initial_dataset_profile is captured once in Phase 1 and never overwritten;
    # dataset_profile is refreshed mid-pipeline for planning purposes and would
    # report the post-split, post-encoding shape here. Fall back to dataset_profile
    # only for older sessions that predate initial_dataset_profile.
    profile = session.initial_dataset_profile or session.dataset_profile or {}
    task = session.task_config or {}

    best = _best_result(session)

    fi = session.feature_importances or {}
    top_feats = fi.get("top_features", [])
    fi_block = ""
    top3_names: list[str] = []
    if top_feats:
        fi_lines = "\n".join(
            f"  {i + 1}. {name} ({pct:.1f}% of total importance)"
            for i, (name, pct) in enumerate(top_feats[:12])
        )
        fi_block = f"\nFEATURE_IMPORTANCES (method: {fi.get('method', 'unknown')}):\n{fi_lines}"
        top3_names = [name for name, _pct in top_feats[:3]]

    if top3_names:
        # The exact strings are repeated with a spelled-out warning because the
        # model kept dropping the category suffix off encoded names, which turns an
        # indicator into a quantity and changes what the sentence claims.
        suffixed = [n for n in top3_names if "_" in n]
        spelling = (
            " Copy each name character for character, including its suffix: write "
            + ", ".join(f'"{n}" not "{n.rsplit("_", 1)[0]}"' for n in suffixed)
            + "."
        ) if suffixed else ""
        takeaway_feature_instruction = (
            f"Open by naming exactly these three, in this order: {', '.join(top3_names)}. "
            f"Do not substitute, reorder, add, or drop any of them.{spelling}"
        )
    else:
        takeaway_feature_instruction = (
            "Per-feature importance was not available for this run. Say so plainly "
            "in one sentence; never guess or invent feature names."
        )

    # Repeated in the schema placeholder because this field was the one the model
    # kept collapsing into a single circular sentence.
    takeaway_hint = (
        (f"name {', '.join(top3_names)} in that order" if top3_names
         else "state that per-feature importance was unavailable")
        + ", then what it means for the outcome that these carry the weight, then "
          "what the run delivers against the user's goal — three sentences, no "
          "closing caveat"
    )

    # The target column name reaches the report only when it means something to a
    # reader. A generic name is replaced in the payload itself rather than
    # forbidden by instruction, which the model ignored.
    raw_target = str(task.get("target_column") or "").strip()
    generic_target = raw_target.lower() in _GENERIC_TARGET_NAMES
    task_for_prompt = dict(task)
    if generic_target:
        task_for_prompt["target_column"] = (
            "<internal placeholder name — describe the outcome in the user's own "
            "words, taken from the goal, and never print this field>"
        )

    eda_sections = _parse_eda_narrative(session.eda_narrative)
    eda_top = _eda_top_features(session, eda_sections)
    # Two different feature rankings are in play and the model kept swapping one
    # for the other, so the contrast is spelled out where the rules are read.
    feature_list_rule = ""
    if top3_names and eda_top:
        feature_list_rule = (
            f"\n- Two feature rankings appear above and they are not "
            f"interchangeable. {', '.join(top3_names)} are what the trained model "
            f"relies on and belong only in \"takeaway\". "
            f"{', '.join(eda_top)} are early exploration figures and belong only in "
            f"\"eda_and_preprocessing\". Never put the exploration names in "
            f"\"takeaway\"."
        )
    eda_narrative_block = ""
    if eda_sections:
        rendered = "\n".join(
            f"  {key.replace('_', ' ')}: {value}" for key, value in eda_sections.items()
        )
        # The names are listed again as an explicit roster because the model kept
        # trailing off into "MaxHR, Oldpeak, and others" when left to read them
        # out of the prose above.
        roster = (
            f"\n  Every feature called most predictive here, in full: {', '.join(eda_top)}. "
            f"If you mention them, list all {len(eda_top)} by name."
            if eda_top else ""
        )
        eda_narrative_block = (
            f"\nEXPLORATION_FINDINGS (measured before modelling, one feature at a "
            f"time):\n{rendered}{roster}\n"
            f"  These findings were written by an earlier stage and carry that "
            f"stage's own hedging. Report only what it measured. Never copy its "
            f"speculative phrasing forward — wording like \"potential issues\", "
            f"\"may impact performance\", or \"could affect the model\" must not "
            f"appear in your output even though it appears above."
        )
    conflicts = _eda_execution_conflicts(eda_top, _mi_dropped_features(decisions))
    # Placed inside the section spec rather than appended as context: as a
    # trailing note the model acknowledged the feature in a list and never
    # explained the disagreement.
    # Split deliberately: the facts go in the data payload, the section spec only
    # points at them. Given ready-made explanatory phrasing the model copied the
    # sentence verbatim without ever attaching it to the feature it was about.
    conflict_fact_block = ""
    conflict_instruction = ""
    conflict_rule = ""
    if conflicts:
        names = ", ".join(conflicts)
        plural = "they were" if len(conflicts) > 1 else "it was"
        conflict_fact_block = (
            f"\nCONTRADICTION_FOUND (two stages disagreed about the same feature):\n"
            f"  - Exploration measured {names} as among the strongest individual "
            f"predictors of the outcome, judged one feature at a time.\n"
            f"  - The modelling stage measured {names} as carrying almost no information "
            f"the other features did not already supply, so {plural} removed before "
            f"training.\n"
            f"  - Both measurements are correct; they answer different questions. The "
            f"trained model reflects the second."
        )
        conflict_instruction = (
            f" It must also account for CONTRADICTION_FOUND above in your own words, "
            f"naming {names} and saying what the reader should conclude."
        )
        # Repeated as a rule as well: stated once inside the field description it
        # was dropped on most runs.
        conflict_rule = (
            f"\n- \"eda_and_preprocessing\" must explain CONTRADICTION_FOUND: say that "
            f"{names} looked strongly predictive on its own but was dropped as redundant "
            f"once the other features were in play. Mentioning {names} only in a list of "
            f"dropped columns does not satisfy this."
        )

    facts_block = _preprocessing_facts(session, decisions)
    fe_block = _feature_engineering_facts(session, decisions)
    scores_block = _model_scores_block(all_results, best)
    error_block = _error_profile_block(all_results, best)

    # task_config["metric"] is set ONLY by the benchmark harness (experiments/benchmark.py)
    # to force one metric across all systems for fair comparison — it is distinct from
    # task_config["evaluation_metric"], Phase 1's adaptive per-dataset choice. When present,
    # the report must describe the metric honestly as fixed for benchmarking, not as an
    # adaptive choice made for this dataset alone.
    forced_metric = str(task.get("metric") or "").strip()
    metric_override_block = (
        f"\nMETRIC_NOTE: this run belongs to a fixed benchmark suite. The metric "
        f"'{forced_metric}' was applied uniformly across all compared systems for "
        f"fair comparison, NOT adaptively selected for this dataset. Say so honestly "
        f"in one clause; do not imply the system chose it from this dataset."
        if forced_metric else ""
    )

    prompt = f"""Write a short, high-signal markdown report on a finished AutoML run.

The reader knows their own domain but is not a data scientist. They are reading
to learn what the system decided and what it found — not to be taught machine
learning. Assume their time is scarce and every sentence must earn its place.

USER_GOAL_AND_CONSTRAINTS:
{task_for_prompt}

INITIAL_DATA_PROFILE:
{condition_profile_for_report(profile, session.eda_results)}

PREPROCESSING_ACTUALLY_APPLIED (verified record — describe these, do not generalise):
{facts_block}

FEATURE_ENGINEERING_RECORD (the only source for the feature engineering section):
{fe_block}

MODEL_SCORES (final round, higher is better):
{scores_block or "  Not recorded."}

BASELINE_AND_ERRORS (measured on the held-out test set):
{error_block or "  Not recorded — say nothing about baselines or error counts."}

DECISIONS_LOG:
{compact_decisions(decisions, limit=60)}{eda_narrative_block}{fi_block}{conflict_fact_block}{metric_override_block}

Return a JSON object with exactly these six string fields, one paragraph each:
"user_request", "task_and_challenges", "eda_and_preprocessing",
"feature_engineering", "model_selection", "takeaway". Every field is mandatory
and must contain real content. Write section headings nowhere — they are added
afterwards. Do not add a summary or conclusion field; there is no such section.

Field by field:

1. "user_request" — the user's goal in their own framing, and what the system
   predicts. Two sentences. No dataset size, no column counts, no metric name —
   those belong in fields 2 and 5.

2. "task_and_challenges" — the task type, then the row and column counts and how
   the columns break down (how many numeric, how many categorical). Then name
   only the characteristics of this data that actually changed a later decision,
   and say what each one changed. Two to four sentences. Do not list individual
   column names. Do not describe preprocessing here; that is field 3. If the data
   had no real difficulties, say that in one clause and move on — a clean dataset
   is a finding, and inventing a challenge to fill the section is a failure.

3. "eda_and_preprocessing" — what exploration measured, then the preprocessing
   that followed, from PREPROCESSING_ACTUALLY_APPLIED: what was dropped and why,
   how missing values and categories were handled, which columns got which
   treatment. Group columns treated the same way.{conflict_instruction} Do not
   cover feature engineering here; that is field 4.

4. "feature_engineering" — from FEATURE_ENGINEERING_RECORD and nothing else. If
   features were created, name them and say what each combines. If none were
   created, say so in one sentence, give the recorded reason only if one appears
   above, and stop. Where the record says no reason is on record, write that one
   sentence and nothing more — do not infer, imply, or speculate about why, and
   do not call it a limitation or an oversight unless the record says so.

5. "model_selection" — name the evaluation metric outright with one clause on
   what it captures, then name all three models in the TOP list above with their
   test scores, in the order given. State the gap between #1 and #2 as a number.
   One clause on why the winning family suits this data. Do not re-rank the list.
   Do not state the baseline or the error counts here — they are appended after
   your text, and repeating them duplicates the line.

6. "takeaway" — exactly three sentences, in this order, and stop after the
   third. Do not append a limitation, a caveat, or a closing thought. (a)
   {takeaway_feature_instruction} (b) What it means for the outcome that these
   particular features carry the weight, in the reader's own domain terms — this
   must say something about the subject matter that the reader did not already
   know from the list itself. "The model uses these to identify at-risk patients"
   and "these carry significant weight" are restatements, not meaning; name what
   kind of measurement they are and what it suggests about the outcome. (c)
   What this run delivers against the user's stated goal, using the error counts
   in the user's own terms — whether the thing they asked for was achieved. The
   caught and missed counts are already printed in field 5, so do not repeat
   them; say what they amount to for the user's stated priority. That sentence
   ends the field and the report. Sentences like "the model uses
   these features to make predictions" or "these are important indicators"
   restate the heading instead of answering it — if you write one, replace it.

Rules — the report is judged on these:
- Fill all six fields. Never merge two into one or leave one empty.
- Write each field as two to four complete sentences. Never pack a field into
  one long sentence held together by commas and "and" — split it up.
- Name the evaluation metric explicitly in "model_selection".
- Name all three top models with their scores in "model_selection". Naming two,
  or naming three that are not the three listed above, is a failure.
- Never invent a difficulty, a risk, or a caveat that the data above does not
  state. "May affect model performance", "some distributional issues", and
  "could introduce bias" are guesses unless a number above supports them. State
  a limitation as the fact it rests on and stop: "the decision threshold was not
  tuned" is a limitation; "which may impact performance in certain scenarios" is
  a guess appended to one, and must be cut.
- Never say a fact is missing, unrecorded, or unexplained. If something is not in
  the data above, leave it out — narrating the gap tells the reader nothing and
  reads as the system not knowing its own run.
- When listing columns, either name every one you mean or give the count. Never
  trail off with "and others" or "such as".
- Say which columns got which treatment. "filled using median or mode" and
  "scaled using standard or robust" are failures — the record above states
  exactly which columns received each, so split them into separate statements.
- End on substance. Never close a field with a line like "the preprocessing
  improved the quality of the data".{feature_list_rule}{conflict_rule}
- Feature importance says how heavily the model uses a feature, not which way
  it pushes the outcome. Never say a higher or lower value raises or lowers
  risk, and never state a direction, threshold, or clinical reading the data
  above does not contain.
- Never call a score gap "significant", "substantial", or "a wide margin" — on
  this scale a few thousandths is a narrow win, so give the number and let the
  reader judge.
- Never write that a finding "did not significantly impact" anything. Cut it.
- Every claim must trace to a name, number, or decision in the data above.
  Never write "an unspecified feature", "a certain column", "several features",
  "and others", "among others", or any similar hedge: the names are all given
  above, so list them or name the ones you mean.
- Never speculate about the system's own reasoning. Where the record above gives
  no reason for a step, say what happened and stop. "The reason is not explicitly
  stated", "it implies that", "presumably", and "this suggests the system" all
  mean you are guessing at your own inputs — report the fact without the guess.
- Use column names exactly as written above, including any suffix. A name like
  "ST_Slope_Up" is an indicator for one category, meaning "ST_Slope is Up" —
  refer to it that way, never as a quantity that can be higher or lower, and
  never shorten it to "ST_Slope".
- Where two columns were treated differently, say which got which. Never write
  "using standard or robust scaling" when the record above says exactly which
  columns received each.
- No textbook definitions. Explain a term only where it changes how a number
  should be read, and then in one clause, not a paragraph. The metric gets one
  clause saying what it captures — never a paragraph defining precision, recall,
  or how an F-score is computed. The reader wants to know what the number means,
  not how it is calculated.
- Do not restate the user's request as if it were a finding, and do not write
  circular sentences ("the model was chosen because it performed best").
- Never print the internal target column name; describe the outcome in the
  user's own words.
- No hyperparameter names or values.
- Do not state the same fact in two fields.
- No markdown headings inside the field text, and no title — the document
  structure is added afterwards.
- Give error counts as whole numbers of cases, exactly as they appear in
  BASELINE_AND_ERRORS. Never round them, never convert a count into a vague
  quantifier like "a small number", and never state a count that block does not
  contain.
- Stay between 350 and 480 words across all six fields combined. Within that
  budget, cutting a weak sentence beats padding a field to look complete — but
  never leave a field thin to save words.

Return JSON in exactly this shape, with these six keys and no others. Do not
rename a key, do not add "introduction" or "conclusion", and make every value a
single plain string. The placeholder in each value says what that value must
contain:
{{
  "user_request": "the user's goal in their framing and what the system predicts",
  "task_and_challenges": "task type, row and column counts, numeric/categorical split, then only what changed a later decision",
  "eda_and_preprocessing": "what exploration measured, then the preprocessing that followed, saying which columns got which treatment",
  "feature_engineering": "what was created and what it combines, or that nothing was created plus the recorded reason if one exists",
  "model_selection": "the metric named outright, all three top models with test scores in order, the gap as a number",
  "takeaway": "{takeaway_hint}"
}}"""

    # Deliberately below LLM_TEMPERATURE_NARRATIVE: at 0.5 the model reworded
    # section headings and reordered content between runs on identical input.
    # The report is a factual record, so fidelity matters more than variety.
    raw = LLMClient().call(
        prompt,
        model=config.LLM_PRO,
        temperature=0.2,
        max_tokens=4096,
        json_mode=True,
    )
    parsed = extract_json_object(raw) if raw else None

    text = ""
    if isinstance(parsed, dict):
        parsed = _resolve_section_keys(parsed)
        rendered: list[str] = []
        empty: list[str] = []
        # The baseline and the error counts are facts, not prose, so they are
        # rendered here rather than asked for. Across runs the model stated the
        # baseline, then dropped it, then replaced it with "a significant
        # improvement over the baseline" — the one part a reader can act on was
        # the part it kept losing. Same reasoning as the code-owned title and
        # headings above; it also removes two rules the prompt was competing over.
        measured = _measured_facts_line(all_results, best)
        for key, heading in _SECTIONS:
            content = str(parsed.get(key) or "").strip()
            # A heading with nothing under it reads worse than a shorter report,
            # so an empty field is dropped and reported rather than rendered.
            if not content:
                empty.append(key)
                continue
            if key == "model_selection" and measured:
                content = f"{content}\n\n{measured}"
            rendered.append(f"{heading}\n{content}")
        if empty:
            logger.warning("report sections returned empty by the model: %s", empty)
        text = "\n\n".join(rendered)

    model_str = best["model"] if best else "N/A"
    score_str = f"{best['test_score']:.4f}" if best and isinstance(best.get("test_score"), (int, float)) else "N/A"
    # Built in code, not by the LLM: title instructions given directly to the
    # LLM were unreliable (it would sometimes omit the title entirely).
    title = (
        f"# Stakeholder Report\n"
        f"*Best model: {model_str} | Test score: {score_str}*\n\n"
    )

    if text:
        # Defensive strip in case the LLM adds a title/subtitle despite being
        # told not to, so the code-generated title above is never duplicated.
        # Only strips an H1 ("# ", single hash) and/or an italic subtitle line
        # — never an H2 ("## ") section heading, which must always survive.
        lines = text.splitlines()
        while lines and (
            (lines[0].strip().startswith("# ") and not lines[0].strip().startswith("## "))
            or lines[0].strip().startswith("*")
        ):
            lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
        body = "\n".join(lines)

        body = _normalize_headings(body)
        if generic_target:
            body = _scrub_generic_target(body, raw_target)
        body = _restore_encoded_names(body, top3_names)
        # Every check is detection-only: the model has every name it needs, so a
        # violation means it skimmed the inputs, and repairing one here would mean
        # inventing the fact it failed to use.
        audit = audit_report_text(
            body,
            expected_models=[name for name, _test, _cv in _ranked_models(all_results)[:3]],
            conflicts=conflicts,
            expected_features=top3_names,
            baseline=_majority_baseline(all_results, best),
        )
        for check, items in audit.items():
            if items:
                logger.warning("report audit — %s: %s", check, items)

        return title + body

    # Deterministic fallback, used when the model returns nothing usable. Built
    # from the same verified blocks as the prompt rather than from generic filler:
    # a short factual record is more use to a reader — and more honest — than a
    # paragraph saying decisions "were applied based on the data profile".
    logger.warning("falling back to the deterministic report; the model returned no usable sections")
    top3 = _ranked_models(all_results)[:3]
    if top3:
        model_lines = "\n".join(
            f"- #{rank} {name}: test {test:.4f}" for rank, (name, test, _cv) in enumerate(top3, 1)
        )
    else:
        model_lines = "- No model scores were recorded."
    return (
        title +
        "## User Request and Task Understanding\n\n"
        f"- Goal: {task.get('goal') or 'Not recorded'}\n"
        f"- Task type: {task.get('task_type') or 'Not recorded'}\n"
        f"- Evaluation metric: {task.get('evaluation_metric') or task.get('metric') or 'Not recorded'}\n"
        + ("" if generic_target else f"- Target column: {raw_target or 'Not recorded'}\n") +
        "\n## Core Task and Data Challenges\n\n"
        f"- Rows: {profile.get('n_rows')}, Columns: {profile.get('n_columns')}\n\n"
        "## EDA Findings and Preprocessing Decisions Applied\n\n"
        f"{facts_block}\n\n"
        "## Feature Engineering\n\n"
        f"{fe_block}\n\n"
        "## Model Selection and Evaluation Metric\n\n"
        f"{model_lines}\n"
        f"- Ran {len(all_results)} execution round(s).\n"
        + (f"{error_block}\n" if error_block else "") +
        "\n"
        "## Takeaway\n\n"
        + (
            "- Most important features: "
            + ", ".join(f"{name} ({pct:.1f}%)" for name, pct in top_feats[:3])
            + "\n"
            if top_feats else "- Per-feature importance was not recorded for this run.\n"
        )
        + "- This report was generated from the recorded run log; the narrative "
          "stage produced no output.\n"
    )
