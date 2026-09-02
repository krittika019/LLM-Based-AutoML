"""Session decision logs in MongoDB (episodic memory) with in-memory fallback."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from automl_system import config

logger = logging.getLogger(__name__)

# When MongoDB is unavailable, entries are stored here so reports still work.
_MEMORY_FALLBACK: dict[str, list[dict[str, Any]]] = {}

# Module-level singleton — avoids one TCP handshake per append_decision call.
_db_available: bool | None = None  # None = not yet probed
_mongo_client: MongoClient | None = None


def _client() -> MongoClient | None:
    global _db_available, _mongo_client
    if _db_available is False:
        return None
    # Return the existing live connection without re-probing.
    if _db_available is True and _mongo_client is not None:
        return _mongo_client
    try:
        c = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=min(config.MONGODB_SERVER_SELECTION_TIMEOUT_MS, 2000),
            connectTimeoutMS=min(config.MONGODB_CONNECT_TIMEOUT_MS, 2000),
        )
        # Force a probe to detect DNS / auth failures immediately
        c.admin.command('ping')
        _db_available = True
        _mongo_client = c
        return c
    except Exception as e:
        if _db_available is not False:
            # Log only on the first failure
            logger.warning("MongoDB connection failed (episodic): %s", e)
            logger.debug("Episodic memory will use in-memory fallback for this process lifetime.")
        _db_available = False
        _mongo_client = None
        return None


def create_session() -> str:
    """Create a new session document and return its session_id."""
    session_id = str(uuid.uuid4())
    _MEMORY_FALLBACK.setdefault(session_id, [])
    client = _client()
    if client is None:
        logger.warning("create_session: no DB; using in-memory decisions log only")
        return session_id
    try:
        db = client[config.DATABASE_NAME]
        coll = db[config.EPISODIC_COLLECTION]
        coll.insert_one(
            {
                "session_id": session_id,
                "entries": [],
                "created_at": datetime.now(timezone.utc),
            }
        )
    except PyMongoError as e:
        logger.warning("create_session failed: %s", e)
    except Exception as e:
        logger.warning("create_session unexpected error: %s", e)
    return session_id


def initialize_session_document(
    session_id: str,
    task_config: dict[str, Any],
    dataset_state_pointer: str,
) -> None:
    """
    Phase 1 Step 3: write the structured session initialization document.

    Stores the initial four memory components:
      - task_config (populated so far, deferred fields marked)
      - dataset_profile (empty dict with reserved keys only — profiling not yet done)
      - decisions_log (empty list with session_id + start_timestamp)
      - dataset_state_pointer (path to the validated raw dataframe)

    MongoDB failure is never pipeline-stopping — logs a warning and continues.
    """
    start_ts = datetime.now(timezone.utc)
    doc: dict[str, Any] = {
        "session_id": session_id,
        "start_timestamp": start_ts,
        "task_config": dict(task_config or {}),
        "dataset_profile": {
            # Reserved keys — will be filled by Step 4 profiling
            "n_rows": None,
            "n_columns": None,
            "columns": {},
            "target_distribution": None,
            "stub": True,
        },
        "decisions_log": [],
        "dataset_state_pointer": dataset_state_pointer,
        "entries": [],
        "created_at": start_ts,
    }
    # Always keep in the in-memory fallback so get_log works without DB
    _MEMORY_FALLBACK.setdefault(session_id, [])

    client = _client()
    if client is None:
        logger.warning(
            "initialize_session_document: MongoDB unavailable — session %s stored in memory only.",
            session_id,
        )
        return
    try:
        db = client[config.DATABASE_NAME]
        coll = db[config.EPISODIC_COLLECTION]
        # Upsert so a prior create_session() call doesn't cause a duplicate-key error
        coll.update_one(
            {"session_id": session_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info("initialize_session_document: session %s written to MongoDB.", session_id)
    except PyMongoError as e:
        logger.warning(
            "initialize_session_document: MongoDB write failed for session %s: %s — continuing with in-memory.",
            session_id, e,
        )
    except Exception as e:
        logger.warning(
            "initialize_session_document: unexpected error for session %s: %s — continuing.",
            session_id, e,
        )


def append_decision(session_id: str, entry: dict[str, Any]) -> None:
    """Append a decision record to the session log."""
    if not config.USE_MEMORY:
        return  # ablation mode — decisions not persisted
    doc = {
        **entry,
        "ts": datetime.now(timezone.utc),
    }
    step = entry.get("step") or entry.get("phase") or "unknown"
    logger.debug("append_decision: step=%s action=%r", step, entry.get("action", "")[:60])

    _MEMORY_FALLBACK.setdefault(session_id, []).append(dict(doc))

    client = _client()
    if client is None:
        logger.debug("append_decision: no DB; entry kept in memory only for session %s", session_id)
        return
    try:
        db = client[config.DATABASE_NAME]
        coll = db[config.EPISODIC_COLLECTION]
        coll.update_one(
            {"session_id": session_id},
            {"$push": {"entries": doc}},
            upsert=True,
        )
    except PyMongoError as e:
        logger.warning("append_decision failed for session %s: %s", session_id, e)
    except Exception as e:
        logger.warning("append_decision unexpected error for session %s: %s", session_id, e)


def get_log(session_id: str) -> list[dict[str, Any]]:
    """Return the ordered list of decision entries for a session."""
    if not config.USE_MEMORY:
        return []  # ablation mode — no memory to retrieve
    client = _client()
    if client is None:
        mem = list(_MEMORY_FALLBACK.get(session_id) or [])
        logger.debug("get_log (memory-only): %d entries for session %s", len(mem), session_id[:8])
        return mem
    try:
        db = client[config.DATABASE_NAME]
        coll = db[config.EPISODIC_COLLECTION]
        doc = coll.find_one({"session_id": session_id})
        if not doc:
            mem = list(_MEMORY_FALLBACK.get(session_id) or [])
            logger.debug("get_log: Mongo miss; returning %d in-memory entries", len(mem))
            return mem
        entries = list(doc.get("entries", []))
        logger.debug("get_log: retrieved %d entries from Mongo for session %s", len(entries), session_id[:8])
        return entries
    except PyMongoError as e:
        logger.warning("get_log failed for session %s: %s", session_id, e)
        mem = list(_MEMORY_FALLBACK.get(session_id) or [])
        logger.debug("get_log (fallback after error): %d entries", len(mem))
        return mem
    except Exception as e:
        logger.warning("get_log unexpected error for session %s: %s", session_id, e)
        return list(_MEMORY_FALLBACK.get(session_id) or [])
