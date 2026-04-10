"""Semantic memory: store run profiles and retrieve similar past runs (numpy cosine, no Atlas)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from automl_system import config

logger = logging.getLogger(__name__)

# Cache availability after first probe — avoids repeated DNS timeouts.
_db_available: bool | None = None


def _client() -> MongoClient | None:
    global _db_available
    if _db_available is False:
        return None
    try:
        c = MongoClient(
            config.MONGODB_URI,
            serverSelectionTimeoutMS=min(config.MONGODB_SERVER_SELECTION_TIMEOUT_MS, 2000),
            connectTimeoutMS=min(config.MONGODB_CONNECT_TIMEOUT_MS, 2000),
        )
        c.admin.command('ping')
        _db_available = True
        return c
    except Exception as e:
        if _db_available is not False:
            logger.warning("MongoDB connection failed (semantic): %s", e)
        _db_available = False
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def store_run(profile_vector: list[float] | np.ndarray, outcome_dict: dict[str, Any]) -> str | None:
    """
    Persist a completed run profile and outcome.
    Returns stored document id or None on failure.
    """
    vec = np.asarray(profile_vector, dtype=np.float64).ravel()
    if vec.size > config.MAX_PROFILE_VECTOR_DIM:
        logger.warning("store_run: vector dim %s exceeds cap", vec.size)
        return None
    client = _client()
    if client is None:
        logger.warning("store_run: no DB; not persisted")
        return None
    doc_id = str(uuid.uuid4())
    try:
        db = client[config.DATABASE_NAME]
        coll = db[config.SEMANTIC_COLLECTION]
        coll.insert_one(
            {
                "_id": doc_id,
                "profile_vector": vec.tolist(),
                "outcome": outcome_dict,
                "created_at": datetime.now(timezone.utc),
            }
        )
        return doc_id
    except PyMongoError as e:
        logger.warning("store_run failed: %s", e)
        return None
    except Exception as e:
        logger.warning("store_run unexpected error: %s", e)
        return None
    finally:
        try:
            client.close()
        except Exception:
            pass


def retrieve_similar(profile_vector: list[float] | np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Return up to top_k most similar past runs by cosine similarity.
    Each item: {"similarity": float, "outcome": dict, "id": str}.
    """
    query = np.asarray(profile_vector, dtype=np.float64).ravel()
    if query.size == 0:
        return []
    client = _client()
    if client is None:
        logger.warning("retrieve_similar: no DB; returning empty list")
        return []
    try:
        db = client[config.DATABASE_NAME]
        coll = db[config.SEMANTIC_COLLECTION]
        cursor = coll.find({})
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in cursor:
            pv = doc.get("profile_vector")
            if not pv:
                continue
            stored = np.asarray(pv, dtype=np.float64).ravel()
            if stored.shape != query.shape:
                continue
            sim = _cosine_similarity(query, stored)
            scored.append(
                (
                    sim,
                    {
                        "similarity": sim,
                        "outcome": doc.get("outcome", {}),
                        "id": str(doc.get("_id", "")),
                    },
                )
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[: max(0, top_k)]]
    except PyMongoError as e:
        logger.warning("retrieve_similar failed: %s", e)
        return []
    except Exception as e:
        logger.warning("retrieve_similar unexpected error: %s", e)
        return []
    finally:
        try:
            client.close()
        except Exception:
            pass
