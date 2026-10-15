"""Run generated Python code in a Docker container, or a local subprocess as fallback."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from automl_system import config

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 20_000
_KILLED_EXIT_CODE = 137  # SIGKILL: from `timeout -s KILL` or the out-of-memory killer

# Only these host variables reach a local subprocess — .env secrets never do.
_SAFE_ENV_KEYS = (
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP",
    "HOME", "USERPROFILE", "LANG", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
)


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    backend: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def _as_text(data: str | bytes | None) -> str:
    if data is None:
        return ""
    return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data


def _clip(text: str) -> str:
    """Keep the tail — tracebacks and result lines are printed last."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return "...[truncated]...\n" + text[-_MAX_OUTPUT_CHARS:]


def _clean_env() -> dict[str, str]:
    env = {k: v for k in _SAFE_ENV_KEYS if (v := os.environ.get(k))}
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    return env


def docker_available() -> bool:
    """True when the Docker daemon is running and the sandbox image is built."""
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", config.SANDBOX_IMAGE],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except Exception:
        return False


# ── Docker Desktop on demand ─────────────────────────────────────────────────
# Docker Desktop's VM holds 1-3 GB of RAM even with no containers. With
# SANDBOX_DOCKER_ON_DEMAND the app starts Docker Desktop only when a run needs the
# sandbox and stops it again afterwards — but only if the app was the one to start it.
_started_docker_desktop = False


def _docker_desktop(*args: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "desktop", *args], capture_output=True, text=True, timeout=timeout)


def docker_desktop_running() -> bool:
    try:
        r = _docker_desktop("status", timeout=20)
        return r.returncode == 0 and "running" in r.stdout.lower()
    except Exception:
        return False


def ensure_docker_started() -> bool:
    """Start Docker Desktop if it is not running and wait until the engine and sandbox
    image are usable. Returns True when Docker is ready."""
    global _started_docker_desktop
    if shutil.which("docker") is None:
        return False
    if docker_desktop_running():
        return docker_available()
    boot = config.SANDBOX_DOCKER_BOOT_TIMEOUT_SEC
    logger.info("Starting Docker Desktop for the sandbox (on demand, up to %ss)", boot)
    t0 = time.perf_counter()
    try:
        _docker_desktop("start", "--timeout", str(boot), timeout=boot + 30)
    except Exception as exc:
        logger.warning("docker desktop start failed: %s", exc)
    _started_docker_desktop = True          # stop it later even if start was slow/failed
    while time.perf_counter() - t0 < boot:
        if docker_available():
            logger.info("Docker Desktop ready after %.0fs", time.perf_counter() - t0)
            return True
        time.sleep(3)
    return False


def stop_docker_if_started() -> None:
    """Stop Docker Desktop if this process started it (frees its VM memory)."""
    global _started_docker_desktop
    if not _started_docker_desktop:
        return
    _started_docker_desktop = False
    logger.info("Stopping Docker Desktop (started on demand for this run)")
    try:
        _docker_desktop("stop", "--detach", timeout=60)
    except Exception as exc:
        logger.warning("docker desktop stop failed: %s", exc)


class Sandbox:
    """One working folder and one execution environment, reused for a whole pipeline run.

    with open_sandbox() as sb:
        sb.put_dataframe("train.csv", df_train)
        res = sb.run(code, timeout=120)
    """

    def __init__(self, backend: str):
        if backend not in ("docker", "local"):
            raise ValueError(f"unknown sandbox backend: {backend!r}")
        self.backend = backend
        self.workdir = Path(tempfile.mkdtemp(prefix="automl_sbx_"))
        self._container: str | None = None
        self._n_runs = 0

    def __enter__(self) -> Sandbox:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.backend != "docker" or self._container:
            return
        name = f"automl-sbx-{uuid.uuid4().hex[:12]}"
        cmd = [
            "docker", "run", "-d", "--rm", "--name", name,
            "--network", "none",
            "--memory", config.SANDBOX_MEMORY, "--memory-swap", config.SANDBOX_MEMORY,
            "--cpus", str(config.SANDBOX_CPUS),
            "--pids-limit", "256",
            "--read-only", "--tmpfs", "/tmp:size=256m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
            "--mount", f"type=bind,source={self.workdir},target=/work",
            config.SANDBOX_IMAGE, "sleep", str(config.SANDBOX_MAX_LIFETIME_SEC),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=config.SANDBOX_START_TIMEOUT_SEC)
        if r.returncode != 0:
            raise RuntimeError(f"could not start sandbox container: {r.stderr.strip()}")
        self._container = name
        logger.info("Sandbox container %s started (workdir=%s)", name, self.workdir)

    def close(self) -> None:
        if self._container:
            subprocess.run(["docker", "rm", "-f", self._container], capture_output=True, timeout=60)
            self._container = None
        shutil.rmtree(self.workdir, ignore_errors=True)
        stop_docker_if_started()

    def _restart(self) -> None:
        if self._container:
            subprocess.run(["docker", "rm", "-f", self._container], capture_output=True, timeout=60)
            self._container = None
        self.start()

    # ── files ──────────────────────────────────────────────────────────────
    def _path(self, name: str) -> Path:
        if Path(name).name != name:
            raise ValueError(f"sandbox file names must be plain names, got {name!r}")
        return self.workdir / name

    def put_dataframe(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self._path(name), index=False)

    def put_file(self, name: str, src: str | Path) -> None:
        shutil.copyfile(src, self._path(name))

    def read_text(self, name: str) -> str | None:
        p = self._path(name)
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None

    # ── execution ──────────────────────────────────────────────────────────
    def run(self, code: str, timeout: int | None = None) -> SandboxResult:
        timeout = int(timeout or config.SANDBOX_TIMEOUT_SEC)
        self._n_runs += 1
        script = f"_run_{self._n_runs}.py"
        self._path(script).write_text(code, encoding="utf-8")

        if self.backend == "docker":
            self.start()
            # `timeout` inside the container kills the script itself; the host-side
            # timeout is only a backstop in case `docker exec` hangs.
            cmd = ["docker", "exec", "-w", "/work", self._container,
                   "timeout", "-s", "KILL", str(timeout), "python", "-W", "ignore", script]
            env, host_timeout = None, timeout + 15
        else:
            cmd = [sys.executable, "-W", "ignore", script]
            env, host_timeout = _clean_env(), timeout

        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                cmd, cwd=self.workdir, env=env, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=host_timeout,
            )
            exit_code, out, err = r.returncode, r.stdout, r.stderr
            timed_out = False
        except subprocess.TimeoutExpired as e:
            exit_code, out, err, timed_out = -1, _as_text(e.stdout), _as_text(e.stderr), True
            if self.backend == "docker":
                self._restart()
        duration = time.perf_counter() - t0

        if self.backend == "docker" and exit_code == _KILLED_EXIT_CODE:
            if duration >= timeout - 1:
                timed_out = True
            else:
                err += f"\nProcess killed: out of memory (sandbox limit {config.SANDBOX_MEMORY})."
        if timed_out:
            err += f"\nProcess killed: exceeded time limit of {timeout}s."

        return SandboxResult(
            exit_code=exit_code, stdout=_clip(out), stderr=_clip(err),
            duration_s=round(duration, 2), timed_out=timed_out, backend=self.backend,
        )


def open_sandbox() -> Sandbox | None:
    """Create a sandbox using SANDBOX_BACKEND; returns None when execution is disabled."""
    choice = config.SANDBOX_BACKEND
    if choice == "off":
        return None
    if choice == "local":
        return Sandbox("local")
    if docker_available():
        return Sandbox("docker")
    if choice == "docker":
        raise RuntimeError(
            f"SANDBOX_BACKEND=docker but Docker is not running or image "
            f"{config.SANDBOX_IMAGE!r} is missing (build it: docker build -t {config.SANDBOX_IMAGE} ./sandbox)"
        )
    logger.warning("Docker sandbox unavailable — falling back to local subprocess")
    return Sandbox("local")


def start_sandbox() -> tuple[Sandbox | None, str]:
    """Open and start a sandbox. If Docker is selected automatically but does not start
    a container in time (e.g. Docker Desktop paused or its VM starved of memory), fall
    back to the local subprocess instead of skipping execution.

    Returns (sandbox or None when disabled, a user-facing note about any fallback).
    """
    on_demand_note = ""
    if (config.SANDBOX_DOCKER_ON_DEMAND and config.SANDBOX_BACKEND in ("auto", "docker")
            and shutil.which("docker") and not docker_available()):
        if not ensure_docker_started():
            on_demand_note = "Docker was not ready in time, so the script ran in a local process."
    sb = open_sandbox()
    if sb is None or sb.backend != "docker":
        return sb, on_demand_note
    try:
        sb.start()
        return sb, ""
    except Exception as exc:
        sb.close()
        if config.SANDBOX_BACKEND == "docker":
            raise
        reason = ("did not start a container within "
                  f"{config.SANDBOX_START_TIMEOUT_SEC}s" if isinstance(exc, subprocess.TimeoutExpired)
                  else "failed to start a container")
        logger.warning("Docker %s — using the local subprocess instead (%s)", reason, exc)
        return Sandbox("local"), f"Docker {reason}, so the script ran in a local process."
