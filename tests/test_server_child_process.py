"""Pipeline runs execute in a child process; the server relays events and survives crashes."""

from __future__ import annotations

import os
import queue
import time

import server


def _crashing_child(events, args):
    """Simulates native code killing the pipeline process mid-run (no Python traceback)."""
    events.put({"type": "phase_start", "phase": 1})
    time.sleep(1)          # let the queue's feeder thread flush, as in a real run
    os._exit(3)


def _drain(run_id: str, timeout: float = 120) -> list[dict]:
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        try:
            ev = server._RUNS[run_id]["q"].get(timeout=1)
        except queue.Empty:
            continue
        out.append(ev)
        if ev.get("type") == "__end__":
            return out
    raise AssertionError(f"no __end__ within {timeout}s: {out}")


def _start(run_id: str, target, args) -> None:
    server._RUNS[run_id] = {"q": queue.Queue(), "script_path": None, "pdf_path": None, "session": None, "done": False}
    events = server._MP.Queue()
    proc = server._MP.Process(target=target, args=(events, args), daemon=True)
    proc.start()
    server.threading.Thread(target=server._relay_child_events, args=(run_id, proc, events), daemon=True).start()


def test_child_run_events_reach_parent():
    run_id = "test-child-ok"
    _start(run_id, server._run_in_child, (run_id, r"Z:\no\such\file.csv", "predict y", None, "x.csv", None))
    types = [e.get("type") for e in _drain(run_id)]
    assert types[-1] == "__end__"
    assert "final" in types or "error" in types
    assert server._RUNS[run_id]["done"]


def test_child_crash_is_reported_and_server_survives():
    run_id = "test-child-crash"
    _start(run_id, _crashing_child, (run_id,))
    events = _drain(run_id)
    assert events[0]["type"] == "phase_start"
    assert any(e.get("type") == "error" and "stopped unexpectedly" in e.get("message", "") for e in events)
    assert events[-1]["type"] == "__end__"
    assert server._RUNS[run_id]["done"]
