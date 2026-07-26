"""Config get/put and schema."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from scan2usd.config import SceneConfig

from scan2usd_gui.bridge import (
    deep_merge,
    load_project,
    load_raw_yaml,
    save_raw_yaml,
    save_yaml_text,
    workspace_summary,
)
from scan2usd_gui.schema import get_schema
from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api", tags=["config"])


def _with_workspace(data: dict[str, Any]) -> dict[str, Any]:
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        cfg = SceneConfig.load(project_state.require_config())
    finally:
        os.chdir(prev)
    data = dict(data)
    data["workspace"] = workspace_summary(cfg, cwd=project_state.cwd)
    return data


@router.get("/schema")
def schema() -> dict[str, Any]:
    return get_schema()


@router.get("/config")
def get_config() -> dict[str, Any]:
    try:
        path = project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _with_workspace(load_project(path, cwd=project_state.cwd))


class ConfigPutBody(BaseModel):
    raw: dict[str, Any] | None = None
    patch: dict[str, Any] | None = None
    yaml_text: str | None = None


@router.put("/config")
def put_config(body: ConfigPutBody) -> dict[str, Any]:
    try:
        path = project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        if body.yaml_text is not None:
            return _with_workspace(save_yaml_text(path, body.yaml_text))
        current = load_raw_yaml(path)
        previous_ws = current.get("workspace_dir")
        if body.raw is not None:
            merged = body.raw
        elif body.patch is not None:
            merged = deep_merge(current, body.patch)
        else:
            raise HTTPException(400, "Provide raw, patch, or yaml_text")
        return _with_workspace(
            save_raw_yaml(
                path,
                merged,
                previous_workspace_dir=(
                    None if previous_ws is None else str(previous_ws)
                ),
            )
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@router.get("/config/value")
def get_resolved() -> dict[str, Any]:
    try:
        path = project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        cfg = SceneConfig.load(path)
    finally:
        os.chdir(prev)
    from scan2usd_gui.bridge import scene_config_to_dict

    return {"config": scene_config_to_dict(cfg)}
