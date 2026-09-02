"""Structured decisions log entries (episodic + session mirror)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from automl_system.memory import episodic
from automl_system.state.session import SessionState


def log_decision(
    session: SessionState,
    step: str,
    action: str,
    rationale: str = "",
    **extra: Any,
) -> None:
    """Append a normalized decision to MongoDB episodic log and session.local_decisions_log."""
    ts = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "step": step,
        "action": action,
        "rationale": rationale or "",
        "timestamp": ts,
        **extra,
    }
    if session.decisions_log_id:
        episodic.append_decision(session.decisions_log_id, entry)
    if not hasattr(session, "local_decisions_log") or session.local_decisions_log is None:
        session.local_decisions_log = []
    session.local_decisions_log.append(dict(entry))


def compact_decisions(entries: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Return the last `limit` decision entries as compact dicts for LLM prompts."""
    out = []
    for e in entries[-limit:]:
        out.append({
            "step": e.get("step") or e.get("phase"),
            "action": e.get("action"),
            "rationale": str(e.get("note") or e.get("rationale") or e.get("text") or "")[:200],
        })
    return out


def merge_logs_for_report(session: SessionState) -> list[dict[str, Any]]:
    """Merge Mongo episodic log with local mirror (dedupe by timestamp+step+action)."""
    remote: list[dict[str, Any]] = []
    if session.decisions_log_id:
        remote = episodic.get_log(session.decisions_log_id) or []
    local = list(session.local_decisions_log or [])
    if not remote and not local:
        return []
    if not remote:
        return local
    if not local:
        return remote
    seen: set[tuple[Any, ...]] = set()
    merged: list[dict[str, Any]] = []
    for e in remote + local:
        key = (
            str(e.get("timestamp") or e.get("ts") or ""),
            str(e.get("step") or e.get("phase") or ""),
            str(e.get("action") or "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(e)
    merged.sort(key=lambda x: str(x.get("timestamp") or x.get("ts") or ""))
    return merged
