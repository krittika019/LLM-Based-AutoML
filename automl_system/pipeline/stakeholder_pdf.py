"""Build a non-technical 1-page stakeholder PDF report using reportlab."""

from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT  # noqa: F401
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    _RL = True
except ImportError:  # pragma: no cover
    _RL = False
    logger.warning(
        "reportlab not installed — stakeholder PDF will be plain-text. "
        "Install with: pip install reportlab>=4.0.0"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _model_score_chart_bytes(all_results: list[dict[str, Any]]) -> bytes | None:
    """Return PNG bytes of a horizontal bar chart of model test scores."""
    scores: list[float] = []
    labels: list[str] = []
    seen: set[str] = set()
    for res in all_results:
        for row in res.get("comparison_table") or []:
            name = (row.get("model_name") or "Model")[:28]
            ts = row.get("test_score")
            if isinstance(ts, (int, float)) and name not in seen:
                scores.append(float(ts))
                labels.append(name)
                seen.add(name)
    if not scores:
        return None

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(scores) * 0.6)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    bar_colors = ["#4ade80" if s == max(scores) else "#60a5fa" for s in scores]
    bars = ax.barh(labels, scores, color=bar_colors, alpha=0.88, edgecolor="none")
    ax.set_xlabel("Score", color="#cbd5e1", fontsize=9)
    ax.set_title("Model Confidence Distribution", color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.invert_yaxis()
    for bar, s in zip(bars, scores):
        ax.text(
            s + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{s:.3f}", va="center", ha="left", color="#f1f5f9", fontsize=8,
        )
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#0f172a")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ── Public API ────────────────────────────────────────────────────────────────


def build_stakeholder_pdf(
    report_data: dict[str, Any],
    all_results: list[dict[str, Any]],
    goal: str = "",
) -> bytes:
    """
    Build a 1-page non-technical stakeholder PDF.
    Falls back to UTF-8 text if reportlab is not installed.
    """
    if not _RL:
        return _plain_text_fallback(report_data, goal)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
    )

    styles = getSampleStyleSheet()
    BLUE = colors.HexColor("#1d4ed8")
    DARK = colors.HexColor("#1e293b")
    MUTED = colors.HexColor("#64748b")

    title_s = ParagraphStyle(
        "TitleS", parent=styles["Heading1"],
        fontSize=20, textColor=BLUE, spaceAfter=2, leading=24,
    )
    goal_s = ParagraphStyle(
        "GoalS", parent=styles["Normal"],
        fontSize=10, textColor=MUTED, spaceAfter=10, leading=13,
    )
    h2_s = ParagraphStyle(
        "H2S", parent=styles["Heading2"],
        fontSize=12, textColor=DARK, spaceBefore=14, spaceAfter=5, leading=16,
    )
    body_s = ParagraphStyle(
        "BodyS", parent=styles["Normal"],
        fontSize=10, leading=15, spaceAfter=4,
    )
    bullet_s = ParagraphStyle(
        "BulletS", parent=styles["Normal"],
        fontSize=10, leading=14, leftIndent=16, spaceAfter=3,
    )

    flowables = []

    # ── Header ────────────────────────────────────────────────────────────────
    flowables.append(Paragraph("AI Model Summary", title_s))
    if goal:
        flowables.append(Paragraph(f"Goal: <i>{goal}</i>", goal_s))
    flowables.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=12))

    # ── 1. What it does ───────────────────────────────────────────────────────
    flowables.append(Paragraph("What Does This Model Do?", h2_s))
    flowables.append(Paragraph(report_data.get("one_sentence") or "—", body_s))

    # ── 2. Confidence ─────────────────────────────────────────────────────────
    flowables.append(Paragraph("How Confident Is It?", h2_s))
    flowables.append(Paragraph(report_data.get("confidence_plain") or "—", body_s))

    # ── Confidence distribution chart ─────────────────────────────────────────
    chart_bytes = _model_score_chart_bytes(all_results)
    if chart_bytes:
        flowables.append(Spacer(1, 0.2 * cm))
        flowables.append(RLImage(io.BytesIO(chart_bytes), width=13 * cm, height=5 * cm))
        flowables.append(Spacer(1, 0.25 * cm))

    # ── 3. Data needed ────────────────────────────────────────────────────────
    flowables.append(Paragraph("What Data Does It Need?", h2_s))
    for item in report_data.get("data_needed") or []:
        flowables.append(Paragraph(f"• {item}", bullet_s))

    # ── 4. Example predictions ────────────────────────────────────────────────
    examples = report_data.get("example_predictions") or []
    if examples:
        flowables.append(Paragraph("Example Predictions", h2_s))
        for ex in examples:
            scenario = ex.get("scenario") or ""
            prediction = ex.get("prediction") or ""
            explanation = ex.get("explanation") or ""
            flowables.append(Paragraph(f"<b>Scenario:</b> {scenario}", bullet_s))
            flowables.append(Paragraph(f"<b>\u2192 {prediction}</b>", bullet_s))
            if explanation:
                flowables.append(Paragraph(f"<i>&nbsp;&nbsp;{explanation}</i>", bullet_s))
            flowables.append(Spacer(1, 0.15 * cm))

    # ── 5. Limitations ────────────────────────────────────────────────────────
    flowables.append(Paragraph("Limitations & When It May Be Wrong", h2_s))
    for lim in report_data.get("limitations") or []:
        flowables.append(Paragraph(f"• {lim}", bullet_s))

    # ── 6. Recommendation ─────────────────────────────────────────────────────
    flowables.append(Paragraph("Recommendation", h2_s))
    flowables.append(Paragraph(report_data.get("recommendation") or "—", body_s))

    doc.build(flowables)
    buf.seek(0)
    return buf.read()


def _plain_text_fallback(report_data: dict[str, Any], goal: str) -> bytes:
    lines = [f"AI Model Summary\n{'='*40}\n", f"Goal: {goal}\n"]
    for key, label in [
        ("one_sentence", "What it does"),
        ("confidence_plain", "How confident"),
        ("recommendation", "Recommendation"),
    ]:
        lines.append(f"\n{label}:\n{report_data.get(key, '—')}\n")
    lines.append("\nData needed:\n")
    for item in report_data.get("data_needed") or []:
        lines.append(f"  - {item}\n")
    lines.append("\nLimitations:\n")
    for lim in report_data.get("limitations") or []:
        lines.append(f"  - {lim}\n")
    return "".join(lines).encode("utf-8")
