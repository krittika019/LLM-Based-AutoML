"""Parse JSON blobs from LLM text responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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
            depth = 0
            for i, ch in enumerate(t[brace:], start=brace):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        t = t[brace : i + 1]
                        break
    try:
        out = json.loads(t)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError as e:
        logger.warning("extract_json_object failed: %s", e)
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
            depth = 0
            for i, ch in enumerate(t[bracket:], start=bracket):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        t = t[bracket : i + 1]
                        break
    try:
        out = json.loads(t)
        return out if isinstance(out, list) else None
    except json.JSONDecodeError as e:
        logger.warning("extract_json_list failed: %s", e)
        return None
