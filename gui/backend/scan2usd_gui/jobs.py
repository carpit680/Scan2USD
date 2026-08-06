"""Background job runner with SSE-friendly log buffering."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobRecord:
    id: str
    command: str
    argv: list[str]
    cwd: str
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    exit_code: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=20_000))
    _proc: subprocess.Popen[str] | None = field(default=None, repr=False)
    _cv: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def append(self, line: str) -> None:
        with self._cv:
            self.lines.append(line)
            self._cv.notify_all()

    def set_status(self, status: str, exit_code: int | None = None) -> None:
        with self._cv:
            self.status = status
            if exit_code is not None:
                self.exit_code = exit_code
            if status == "running" and self.started_at is None:
                self.started_at = time.time()
            if status in {"succeeded", "failed", "cancelled"}:
                self.finished_at = time.time()
            self._cv.notify_all()

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "status": self.status,
            "exit_code": self.exit_code,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "line_count": len(self.lines),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [j.snapshot() for j in sorted(self._jobs.values(), key=lambda x: x.created_at, reverse=True)]

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def start(
        self,
        *,
        command: str,
        argv: list[str],
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        job = JobRecord(id=job_id, command=command, argv=argv, cwd=str(cwd))
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job, env), daemon=True)
        thread.start()
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        proc = job._proc
        if proc is None or job.status not in {"queued", "running"}:
            return job
        job.set_status("cancelled", exit_code=-1)
        job.append("[gui] cancel requested — stopping process group")

        def _kill(sig: int) -> None:
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    if sig == signal.SIGTERM:
                        proc.terminate()
                    else:
                        proc.kill()
                except Exception:  # noqa: BLE001
                    pass

        _kill(signal.SIGTERM)

        def _force_kill() -> None:
            if proc.poll() is None:
                job.append("[gui] still running — sending SIGKILL")
                _kill(signal.SIGKILL)

        threading.Timer(2.0, _force_kill).start()
        return job

    def _run(self, job: JobRecord, env: dict[str, str] | None) -> None:
        job.set_status("running")
        job.append(f"$ {' '.join(job.argv)}")
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        run_env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.Popen(
                job.argv,
                cwd=job.cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            job._proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                job.append(line.rstrip("\n"))
            code = proc.wait()
            if job.status == "cancelled":
                return
            job.set_status("succeeded" if code == 0 else "failed", exit_code=code)
            if code == 2:
                job.append(
                    "[gui] exit code 2 — often ReviewRequired; open Review to approve gates"
                )
            job.append(f"[gui] process exited with code {code}")
        except Exception as exc:  # noqa: BLE001
            job.append(f"[gui] failed to start: {exc}")
            job.set_status("failed", exit_code=-1)


job_manager = JobManager()


def _append_option_flags(argv: list[str], options: dict[str, Any], *, bool_no_keys: set[str]) -> None:
    for key in sorted(options):
        if key.startswith("_pos_"):
            val = options[key]
            if val is None or val == "":
                continue
            argv.append(str(val))

    for key, value in options.items():
        if key.startswith("_"):
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key in bool_no_keys:
                argv.append(f"--no-{key.replace('_', '-')}")
            continue
        if value is None or value == "":
            continue
        argv.extend([flag, str(value)])


def resolve_scan2usd_argv(command: str, config_path: Path, options: dict[str, Any]) -> list[str]:
    """Build ``python -m scan2usd.cli <command> <config> …`` argv from option dict."""
    import shutil
    import sys

    scan2usd_bin = shutil.which("scan2usd")
    if scan2usd_bin:
        argv = [scan2usd_bin, command, str(config_path)]
    else:
        argv = [sys.executable, "-m", "scan2usd.cli", command, str(config_path)]

    _append_option_flags(
        argv,
        options,
        bool_no_keys={
            "boxes",
            "ultralytics_runs",
            "isaac",
            "force",
            "dry_run",
            "yes",
            "share",
            "viewer",
            "skip_train",
            "skip_process_data",
            "skip_render",
            "downloads",
        },
    )
    return argv


def resolve_tool_argv(
    tool_id: str,
    options: dict[str, Any],
    *,
    external: dict[str, str] | None = None,
) -> list[str]:
    """Build allowlisted ``python <tools/…> …`` argv. Rejects unknown tool ids."""
    import sys

    from scan2usd_gui.paths import repo_root
    from scan2usd_gui.schema import TOOL_DEFS

    tool = next((t for t in TOOL_DEFS if t["id"] == tool_id), None)
    if tool is None:
        raise KeyError(tool_id)

    script_rel = str(tool["script"])
    # Hard allowlist: script must be under tools/ and match the registered path
    if not script_rel.startswith("tools/") or ".." in Path(script_rel).parts:
        raise ValueError(f"Refusing non-allowlisted script: {script_rel}")
    # Anchored to the checkout, not the project cwd: tools/ ships with the code,
    # while the project cwd is wherever the user's scene data lives.
    root = repo_root()
    script_path = (root / script_rel).resolve()
    tools_root = (root / "tools").resolve()
    if not str(script_path).startswith(str(tools_root) + "/") and script_path != tools_root:
        raise ValueError(f"Script escapes tools/: {script_path}")
    if not script_path.is_file():
        raise FileNotFoundError(f"Tool script not found: {script_path}")

    ext = external or {}
    python_from = tool.get("python_from")
    if python_from:
        python_bin = str(ext.get(python_from) or "").strip() or None
    else:
        python_bin = None
    if not python_bin:
        python_bin = sys.executable

    argv = [python_bin, str(script_path)]
    _append_option_flags(
        argv,
        options,
        bool_no_keys={"ground_marker"},
    )
    return argv
