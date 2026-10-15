"""Docker Desktop is started only when a run needs the sandbox, and stopped afterwards —
but only when the app itself started it. Docker commands are faked here."""

from __future__ import annotations

import pytest

from automl_system import config
from automl_system.sandbox import runner


@pytest.fixture
def fake_docker(monkeypatch):
    calls: list[tuple] = []
    state = {"running": False, "ready_after": 1, "checks": 0}

    def desktop(*args, timeout):
        calls.append(args)
        if args[0] == "start":
            state["running"] = True
        if args[0] == "stop":
            state["running"] = False

    def available():
        if not state["running"]:
            return False
        state["checks"] += 1
        return state["checks"] >= state["ready_after"]

    monkeypatch.setattr(runner, "_docker_desktop", desktop)
    monkeypatch.setattr(runner, "docker_desktop_running", lambda: state["running"])
    monkeypatch.setattr(runner, "docker_available", available)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "docker")
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "_started_docker_desktop", False)
    monkeypatch.setattr(config, "SANDBOX_DOCKER_ON_DEMAND", True)
    monkeypatch.setattr(config, "SANDBOX_BACKEND", "auto")
    return calls, state


def test_starts_when_needed_and_stops_after(fake_docker):
    calls, state = fake_docker
    assert runner.ensure_docker_started()
    assert calls[0][0] == "start"
    runner.stop_docker_if_started()
    assert calls[-1][:2] == ("stop", "--detach") and not state["running"]
    runner.stop_docker_if_started()                      # second call is a no-op
    assert [c[0] for c in calls].count("stop") == 1


def test_leaves_user_started_docker_alone(fake_docker):
    calls, state = fake_docker
    state["running"] = True                              # user opened Docker Desktop
    assert runner.ensure_docker_started()
    runner.stop_docker_if_started()
    assert calls == []                                   # never started, never stopped


def test_slow_boot_falls_back_to_local_and_still_stops(fake_docker, monkeypatch):
    calls, state = fake_docker
    state["ready_after"] = 10 ** 9                       # engine never becomes ready
    monkeypatch.setattr(config, "SANDBOX_DOCKER_BOOT_TIMEOUT_SEC", 0)
    sb, note = runner.start_sandbox()
    assert sb.backend == "local" and "local process" in note
    sb.close()
    assert [c[0] for c in calls] == ["start", "stop"]
