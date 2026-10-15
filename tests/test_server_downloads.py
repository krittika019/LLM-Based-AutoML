"""Script/split downloads survive a lost run registry (e.g. after a backend restart)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import server


def test_artifact_found_without_registry():
    run_id = str(uuid.uuid4())
    path = Path(tempfile.gettempdir()) / f"automl_pipeline_{run_id}.py"
    path.write_text("print('hi')", encoding="utf-8")
    try:
        assert run_id not in server._RUNS
        assert server._run_artifact(run_id, "script_path", path.name) == str(path)
    finally:
        path.unlink()


def test_artifact_rejects_non_uuid_run_ids():
    assert server._run_artifact("../../secret", "script_path", "x.py") is None
