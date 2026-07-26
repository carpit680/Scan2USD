"""Project open / status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scan2usd.config import SceneConfig

from scan2usd_gui.bridge import load_project, workspace_summary
from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/project", tags=["project"])


class OpenProjectBody(BaseModel):
    config_path: str
    cwd: str | None = None


@router.get("")
def get_project() -> dict:
    if project_state.config_path is None:
        return {"loaded": False, "config_path": None, "cwd": str(project_state.cwd)}
    try:
        data = load_project(project_state.config_path, cwd=project_state.cwd)
        import os

        prev = Path.cwd()
        try:
            os.chdir(project_state.cwd)
            cfg = SceneConfig.load(project_state.config_path)
        finally:
            os.chdir(prev)
        data["loaded"] = True
        data["workspace"] = workspace_summary(cfg, cwd=project_state.cwd)
        return data
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@router.put("")
def open_project(body: OpenProjectBody) -> dict:
    path = Path(body.config_path).expanduser()
    if not path.is_file():
        raise HTTPException(404, f"Config not found: {path}")
    cwd = Path(body.cwd).expanduser() if body.cwd else Path.cwd()
    try:
        data = load_project(path, cwd=cwd)
        import os

        prev = Path.cwd()
        try:
            os.chdir(cwd)
            cfg = SceneConfig.load(path)
        finally:
            os.chdir(prev)
        data["loaded"] = True
        data["workspace"] = workspace_summary(cfg, cwd=cwd)
        return data
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


class CreateProjectBody(BaseModel):
    path: str
    template: str = Field(default="configs/example_scene.yaml")
    cwd: str | None = None


@router.post("/create")
def create_project(body: CreateProjectBody) -> dict:
    dest = Path(body.path).expanduser().resolve()
    if dest.exists():
        raise HTTPException(400, f"Already exists: {dest}")
    template = Path(body.template).expanduser()
    if not template.is_file():
        # try relative to repo / cwd
        cand = Path.cwd() / body.template
        if cand.is_file():
            template = cand
        else:
            raise HTTPException(404, f"Template not found: {body.template}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(template.read_text())
    cwd = Path(body.cwd).expanduser() if body.cwd else Path.cwd()
    return open_project(OpenProjectBody(config_path=str(dest), cwd=str(cwd)))
