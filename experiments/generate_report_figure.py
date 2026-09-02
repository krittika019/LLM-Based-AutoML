"""
Render the actual generated stakeholder report from a case-study run as a
document-style figure for the qualitative case study section of the paper.

Reads experiments/results/case_study_<dataset>.json ("report" field) and
renders it verbatim — no text is rewritten or invented here, this is a
facsimile of real system output.

Usage:
    python experiments/generate_report_figure.py --dataset heart
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = Path(__file__).parent / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

TITLE_COLOR  = "#1f3a5f"
HEADER_COLOR = "#1f3a5f"
RULE_COLOR   = "#5b8dd9"
BODY_COLOR   = "#2b2b2b"
BORDER_COLOR = "#999999"
BADGE_BG     = "#eef3fb"
TABLE_BG     = "#f4f7fc"

W = 7.6           # figure width, inches
MARGIN = 0.35
CONTENT_W = W - 2 * MARGIN
HEADER_FS = 9.5
BODY_FS = 8.3
LINE_H = 0.155
SECTION_GAP = 0.16
WRAP_CHARS = 98


def parse_sections(md: str) -> list[tuple[str, list[str]]]:
    """Split the report into (heading, body) pairs.

    The document's H1 and its italic subtitle are dropped: the figure draws its
    own title block from the JSON, so keeping them renders the title twice.
    """
    sections: list[tuple[str, list[str]]] = []
    cur_title = None
    cur_lines: list[str] = []
    preamble: list[str] = []
    for raw in md.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            if cur_title is not None:
                sections.append((cur_title, cur_lines))
            cur_lines = []
            cur_title = None if len(m.group(1)) == 1 else m.group(2).strip().strip("*").strip()
        elif cur_title is not None:
            cur_lines.append(line)
        elif line.strip() and not sections:
            preamble.append(line)
    if cur_title is not None:
        sections.append((cur_title, cur_lines))
    # Only genuine pre-heading content becomes an Overview block; the H1's own
    # subtitle line never reaches here because the H1 clears cur_title.
    preamble = [l for l in preamble if not l.strip().startswith("*")]
    if preamble:
        sections.insert(0, ("Overview", preamble))
    return sections


def extract_table(body_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Pulls a markdown table out of a section body, returns (headers, rows)."""
    table_lines = [l for l in body_lines if l.strip().startswith("|")]
    headers: list[str] = []
    rows: list[list[str]] = []
    for l in table_lines:
        cells = [c.strip().strip("`") for c in l.strip().strip("|").split("|")]
        if all(re.fullmatch(r"-+", c) for c in cells if c):
            continue
        if not headers:
            headers = cells
        else:
            rows.append(cells)
    return headers, rows


def _row_wrapped_params(row: list[str], wrap_chars: int = 78) -> list[str]:
    """Wraps a table row's second cell (params) — shared by both the height-
    measuring pass and the drawing pass so they never disagree on line count."""
    if len(row) <= 1 or not row[1]:
        return []
    return textwrap.wrap(row[1], wrap_chars)


def _table_block_height(rows: list[list[str]]) -> float:
    """Total height (in the same LINE_H units the drawing pass advances `ty`
    by) a table will occupy — shared by both passes so the background patch
    and total figure height always match what actually gets drawn."""
    if not rows:
        return 0.0
    units = 0.55 + 1.3  # header offset + header row
    for row in rows:
        wrapped = _row_wrapped_params(row)
        units += (len(wrapped) + 0.3) if wrapped else 1.3
    return LINE_H * units


def render_report_figure(dataset: str) -> None:
    json_path = RESULTS / f"case_study_{dataset}.json"
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    report_md = raw["report"]
    best_model = raw.get("best_model", "")
    best_score = raw.get("best_score")

    sections = parse_sections(report_md)

    # ── First pass: measure total height needed ────────────────────────────
    fig = plt.figure(figsize=(W, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.axis("off")

    cursor = 0.0
    blocks = []  # (kind, payload, height)

    # Title block
    title_h = 0.85
    blocks.append(("title", None, title_h))
    cursor += title_h

    for title, body_lines in sections:
        headers, rows = extract_table(body_lines)
        prose_lines = [l for l in body_lines if l.strip() and not l.strip().startswith("|")]
        prose = " ".join(l.strip() for l in prose_lines).strip()
        # matplotlib draws plain strings, so inline markdown emphasis would render
        # as literal asterisks. Stripped rather than styled: this figure is a
        # facsimile of the report's text, and a bold run mid-paragraph is not worth
        # splitting the line into separate styled draws.
        prose = re.sub(r"\*{1,3}(?=\S)(.+?)(?<=\S)\*{1,3}", r"\1", prose)
        wrapped = textwrap.wrap(prose, WRAP_CHARS) if prose else []

        h = LINE_H * 1.3          # header line
        h += LINE_H * len(wrapped)
        if rows:
            h += LINE_H * 0.35 + _table_block_height(rows)
        h += SECTION_GAP
        blocks.append(("section", (title, wrapped, headers, rows), h))
        cursor += h

    total_h = cursor + MARGIN * 2
    plt.close(fig)

    # ── Second pass: actually draw ──────────────────────────────────────────
    fig = plt.figure(figsize=(W, total_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()
    ax.axis("off")

    # Card background + border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.05), W - 0.1, total_h - 0.1,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, edgecolor=BORDER_COLOR, facecolor="white", zorder=1))

    y = MARGIN

    # Title
    ax.text(MARGIN, y, "Stakeholder Report",
             fontsize=12.5, fontweight="bold", color=TITLE_COLOR,
             ha="left", va="top", zorder=3)
    y += 0.32
    subtitle = f"Dataset: {dataset}   |   Best model: {best_model}"
    if isinstance(best_score, (int, float)):
        subtitle += f"   |   Test score: {best_score:.4f}"
    ax.text(MARGIN, y, subtitle, fontsize=8.5, color="#555555",
             ha="left", va="top", style="italic", zorder=3)
    y += 0.30
    ax.plot([MARGIN, W - MARGIN], [y, y], color=RULE_COLOR, linewidth=1.4, zorder=3)
    y += 0.22

    for title, wrapped, headers, rows in (b[1] for b in blocks if b[0] == "section"):
        ax.text(MARGIN, y, title, fontsize=HEADER_FS, fontweight="bold",
                color=HEADER_COLOR, ha="left", va="top", zorder=3)
        y += LINE_H * 1.3

        for line in wrapped:
            ax.text(MARGIN, y, line, fontsize=BODY_FS, color=BODY_COLOR,
                     ha="left", va="top", zorder=3)
            y += LINE_H

        if rows:
            y += LINE_H * 0.35
            table_h = _table_block_height(rows)
            ax.add_patch(mpatches.FancyBboxPatch(
                (MARGIN, y), CONTENT_W, table_h,
                boxstyle="round,pad=0.01,rounding_size=0.04",
                linewidth=0.8, edgecolor=RULE_COLOR, facecolor=TABLE_BG, zorder=2))
            col_x = MARGIN + 0.15
            col2_x = MARGIN + 1.6
            ty = y + LINE_H * 0.55
            ax.text(col_x, ty, headers[0], fontsize=BODY_FS, fontweight="bold",
                    color=HEADER_COLOR, ha="left", va="top", zorder=3)
            if len(headers) > 1:
                ax.text(col2_x, ty, headers[1], fontsize=BODY_FS, fontweight="bold",
                        color=HEADER_COLOR, ha="left", va="top", zorder=3)
            ty += LINE_H * 1.3
            for row in rows:
                ax.text(col_x, ty, row[0], fontsize=BODY_FS, color=BODY_COLOR,
                        ha="left", va="top", zorder=3)
                wrapped_params = _row_wrapped_params(row)
                if wrapped_params:
                    for pl in wrapped_params:
                        ax.text(col2_x, ty, pl, fontsize=BODY_FS - 0.3, color=BODY_COLOR,
                                ha="left", va="top", family="monospace", zorder=3)
                        ty += LINE_H
                    ty += LINE_H * 0.3
                else:
                    ty += LINE_H * 1.3
            y += table_h

        y += SECTION_GAP

    out_png = FIGS / "fig2_sample_report.png"
    out_pdf = FIGS / "fig2_sample_report.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="heart")
    args = parser.parse_args()
    render_report_figure(args.dataset)
