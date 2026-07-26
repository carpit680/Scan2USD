"""Job start / stream / cancel."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scan2usd.config import SceneConfig

from scan2usd_gui.bridge import apply_command_path_defaults, load_raw_yaml
from scan2usd_gui.jobs import job_manager, resolve_scan2usd_argv, resolve_tool_argv
from scan2usd_gui.schema import COMMAND_DEFS, TOOL_DEFS
from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

_COMMAND_IDS = {c["id"] for c in COMMAND_DEFS}
_TOOL_IDS = {t["id"] for t in TOOL_DEFS}
_ALL_IDS = _COMMAND_IDS | _TOOL_IDS


class StartJobBody(BaseModel):
    command: str
    options: dict[str, Any] = Field(default_factory=dict)


@router.get("")
def list_jobs() -> dict:
    return {"jobs": job_manager.list_jobs()}


@router.post("")
def start_job(body: StartJobBody) -> dict:
    if body.command not in _ALL_IDS:
        raise HTTPException(400, f"Unknown command: {body.command}")
    try:
        config_path = project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

    options = dict(body.options)
    try:
        prev = Path.cwd()
        try:
            os.chdir(project_state.cwd)
            cfg = SceneConfig.load(config_path)
        finally:
            os.chdir(prev)
        options = apply_command_path_defaults(
            body.command, options, cfg, cwd=project_state.cwd
        )
        if body.command in _TOOL_IDS:
            raw = load_raw_yaml(config_path)
            external = raw.get("external") if isinstance(raw.get("external"), dict) else {}
            argv = resolve_tool_argv(
                body.command,
                options,
                cwd=project_state.cwd,
                external={str(k): str(v) for k, v in external.items()},
            )
        else:
            argv = resolve_scan2usd_argv(body.command, config_path, options)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc

    job = job_manager.start(command=body.command, argv=argv, cwd=project_state.cwd)
    return job.snapshot()


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return job_manager.get(job_id).snapshot()
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    try:
        return job_manager.cancel(job_id).snapshot()
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc


@router.get("/{job_id}/logs")
def get_job_logs(job_id: str, after: int = 0) -> dict:
    try:
        job = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc
    lines = list(job.lines)
    after = max(0, after)
    chunk = lines[after:]
    snap = job.snapshot()
    snap["lines"] = chunk
    snap["next"] = after + len(chunk)
    return snap


@router.get("/{job_id}/events")
async def job_events(job_id: str, after: int = 0) -> StreamingResponse:
    try:
        job = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc

    async def gen():
        idx = max(0, int(after))
        while True:
            # Snapshot under the lock; never Condition.wait() from another thread
            # without holding the lock (that crashed SSE with RuntimeError).
            with job._cv:
                batch = list(job.lines)[idx:]
                status = job.status
                exit_code = job.exit_code
                total = len(job.lines)
            for line in batch:
                idx += 1
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"
            if status not in {"queued", "running"} and idx >= total:
                yield f"data: {json.dumps({'type': 'done', 'status': status, 'exit_code': exit_code})}\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")
