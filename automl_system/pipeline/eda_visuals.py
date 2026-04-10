"""Matplotlib EDA plots as base64 PNG for Gradio chat (markdown images)."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl_system.llm.client import LLMClient
from automl_system.state.session import SessionState
from automl_system.utils.llm_json import extract_json_object

_FACE = "#1a1a1a"
_TEXT = "#e8e8e8"


def _close_fig(fig) -> None:
    plt.close(fig)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=_FACE, dpi=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _md_image(b64: str, alt: str = "EDA Plot") -> str:
    return f"![{alt}](data:image/png;base64,{b64})"


def plot_target_distribution(df: pd.DataFrame, target: str) -> str | None:
    if target not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(_FACE)
    ax.set_facecolor(_FACE)
    vc = df[target].value_counts(dropna=False)
    ax.bar([str(x) for x in vc.index], vc.values, color="#4a9eff", edgecolor="#333")
    ax.set_title("Target distribution", color=_TEXT)
    ax.set_xlabel(str(target), color=_TEXT)
    ax.set_ylabel("Count", color=_TEXT)
    ax.tick_params(colors=_TEXT)
    for spine in ax.spines.values():
        spine.set_color("#444")
    b64 = _fig_to_b64(fig)
    _close_fig(fig)
    return _md_image(b64, "Target distribution")


def plot_missing_heatmap(df: pd.DataFrame, max_rows: int = 120) -> str | None:
    null_cols = [c for c in df.columns if df[c].isna().any()]
    if not null_cols:
        return None
    sub = df[null_cols].head(max_rows)
    mat = sub.isna().astype(float).values
    fig, ax = plt.subplots(figsize=(max(5, len(null_cols) * 0.6), 2.8))
    fig.patch.set_facecolor(_FACE)
    ax.set_facecolor(_FACE)
    im = ax.imshow(mat.T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(null_cols)))
    ax.set_yticklabels([str(c)[:40] for c in null_cols], fontsize=8, color=_TEXT)
    ax.set_xlabel(f"Row index (first {len(sub)} rows)", color=_TEXT)
    ax.set_title("Missing values (yellow = missing)", color=_TEXT)
    ax.tick_params(colors=_TEXT)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    b64 = _fig_to_b64(fig)
    _close_fig(fig)
    return _md_image(b64, "Missing values heatmap")


def _numeric_corr_with_target(df: pd.DataFrame, target: str) -> list[tuple[str, float]]:
    if target not in df.columns:
        return []
    t = df[target]
    if pd.api.types.is_numeric_dtype(t):
        ty = t
    else:
        ty = pd.Categorical(t.astype(str)).codes
        if len(np.unique(ty)) < 2:
            return []
    out: list[tuple[str, float]] = []
    for c in df.columns:
        if c == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        sub = pd.DataFrame({"x": df[c], "y": ty}).dropna()
        if len(sub) < 3:
            continue
        r = float(np.corrcoef(sub["x"], sub["y"])[0, 1])
        if not np.isnan(r):
            out.append((c, r))
    out.sort(key=lambda x: -abs(x[1]))
    return out[:15]


def plot_correlation_heatmap(df: pd.DataFrame, target: str) -> str | None:
    top = _numeric_corr_with_target(df, target)
    if len(top) < 2:
        return None
    cols = [c for c, _ in top[:15]]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    if sub.shape[1] < 2:
        return None
    cmat = sub.corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor(_FACE)
    ax.set_facecolor(_FACE)
    im = ax.imshow(cmat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cmat.columns)))
    ax.set_yticks(range(len(cmat.index)))
    ax.set_xticklabels([str(x)[:20] for x in cmat.columns], rotation=45, ha="right", fontsize=7, color=_TEXT)
    ax.set_yticklabels([str(x)[:20] for x in cmat.index], fontsize=7, color=_TEXT)
    ax.set_title("Correlation heatmap (top features vs target-driven set)", color=_TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    b64 = _fig_to_b64(fig)
    _close_fig(fig)
    return _md_image(b64, "Correlation heatmap")


def plot_top_numeric_distributions(df: pd.DataFrame, target: str, k: int = 5) -> str | None:
    ranked = _numeric_corr_with_target(df, target)
    cols = [c for c, _ in ranked[:k]]
    if not cols:
        num = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        cols = num[:k]
    if not cols:
        return None
    n = min(len(cols), 4)  # cap at 4 subplots
    cols = cols[:n]
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.4))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(_FACE)
    for ax, c in zip(axes, cols):
        ax.set_facecolor(_FACE)
        s = df[c].dropna()
        if len(s) < 2:
            continue
        ax.hist(s, bins=min(30, max(10, len(s) // 5)), density=True, color="#6a7a8a", alpha=0.75)
        try:
            s.plot.kde(ax=ax, color="#ffb347", linewidth=2)
        except Exception:
            pass
        ax.set_title(str(c)[:28], fontsize=9, color=_TEXT)
        ax.tick_params(colors=_TEXT, labelsize=7)
    fig.suptitle("Top numeric features: histogram + KDE", color=_TEXT, fontsize=11)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    _close_fig(fig)
    return _md_image(b64, "Numeric distributions")


def _batched_plot_captions(session: SessionState, plot_labels: list[str], eda_compact: dict[str, Any]) -> list[str]:
    if not plot_labels:
        return []
    client = LLMClient()
    prompt = f"""You are explaining EDA plots for a tabular ML task.

GOAL / TARGET: {session.task_config.get("goal", "")}
TARGET COLUMN: {session.task_config.get("target_column", "")}

EDA SUMMARY (JSON):
{eda_compact}

PLOTS (in order): {plot_labels}

For EACH plot, write 2-3 sentences in plain English: what the plot shows and why it matters for modeling.
Respond with JSON only:
{{"captions": ["...", "...", ...]}}
The captions array MUST have exactly {len(plot_labels)} strings in the same order as the plots."""
    raw = client.call(prompt)
    parsed = extract_json_object(raw) if raw else None
    caps: list[str] = []
    if parsed and isinstance(parsed.get("captions"), list):
        caps = [str(x) for x in parsed["captions"]]
    while len(caps) < len(plot_labels):
        caps.append(f"This chart ({plot_labels[len(caps)]}) summarizes patterns relevant to the target.")
    return caps[: len(plot_labels)]


def build_eda_chat_messages(df: pd.DataFrame, session: SessionState, eda_compact: dict[str, Any]) -> list[dict[str, Any]]:
    """Build assistant chat messages: one image markdown per plot, then caption messages."""
    target = str((session.task_config or {}).get("target_column") or "")
    out: list[dict[str, Any]] = []
    labels: list[str] = []

    p1 = plot_target_distribution(df, target) if target else None
    if p1:
        out.append({"role": "assistant", "content": p1})
        labels.append("Target class balance (bar chart)")

    p2 = plot_missing_heatmap(df)
    if p2:
        out.append({"role": "assistant", "content": p2})
        labels.append("Missing values heatmap")

    p3 = plot_correlation_heatmap(df, target) if target else None
    if p3:
        out.append({"role": "assistant", "content": p3})
        labels.append("Feature correlation heatmap")

    p4 = plot_top_numeric_distributions(df, target)
    if p4:
        out.append({"role": "assistant", "content": p4})
        labels.append("Top numeric histograms with KDE")

    if not labels:
        return out

    caps = _batched_plot_captions(session, labels, eda_compact)
    for lab, cap in zip(labels, caps, strict=False):
        out.append(
            {
                "role": "assistant",
                "content": f"**{lab}**\n\n{cap}",
            }
        )

    ranked = _numeric_corr_with_target(df, target)
    top3 = ranked[:3]
    imb = session.eda_results.get("target_imbalance") or {}
    summary_lines = [
        "### EDA summary",
        f"- **Top 3 features by |correlation| with target:** "
        + (", ".join(f"`{a}` ({b:+.3f})" for a, b in top3) if top3 else "n/a (non-numeric target or insufficient numeric features)"),
        f"- **Class imbalance:** severity **{imb.get('severity', 'unknown')}** (top class share {imb.get('top_class_share', 'n/a')}).",
        "- Review skewed distributions and missingness above before trusting hold-out scores.",
    ]
    out.append({"role": "assistant", "content": "\n".join(summary_lines)})
    return out
