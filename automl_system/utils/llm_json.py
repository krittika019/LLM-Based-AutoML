"""Parse JSON blobs from LLM text responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _find_balanced_end(t: str, start: int, open_ch: str, close_ch: str) -> int | None:
    """Finds the index of the char closing the bracket opened at `start`,
    tracking JSON string literals (and escapes within them) so a stray
    open_ch/close_ch inside a free-text field value doesn't throw off the
    depth count. Returns None if the text ends before the bracket closes."""
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(t[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a single JSON object from model output (markdown fences optional)."""
    if not text or not text.strip():
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t)
    if m:
        t = m.group(1)
    else:
        brace = t.find("{")
        if brace != -1:
            end = _find_balanced_end(t, brace, "{", "}")
            if end is not None:
                t = t[brace : end + 1]
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError as e:
        logger.debug("extract_json_object failed: %s", e)
        return None


def extract_json_list(text: str) -> list[Any] | None:
    """Extract a JSON array from model output."""
    if not text or not text.strip():
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", t)
    if m:
        t = m.group(1)
    else:
        bracket = t.find("[")
        if bracket != -1:
            end = _find_balanced_end(t, bracket, "[", "]")
            if end is not None:
                t = t[bracket : end + 1]
    try:
        out = json.loads(t)
        return out if isinstance(out, list) else None
    except json.JSONDecodeError as e:
        logger.debug("extract_json_list failed: %s", e)
        return None
