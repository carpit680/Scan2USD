"""Workspace artifact browser."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from scan2usd.config import SceneConfig

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])

_TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".txt", ".md", ".csv", ".log", ".usda"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _cfg() -> SceneConfig:
    path = project_state.require_config()
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        return SceneConfig.load(path)
    finally:
        os.chdir(prev)


def _roots(cfg: SceneConfig) -> dict[str, Path]:
    return {
        "workspace": Path(cfg.workspace_dir).resolve(),
        "frames": Path(cfg.frames_dir).resolve(),
        "masks": Path(cfg.segmentation.masks_dir or (cfg.workspace_dir / "masks")).resolve(),
        "usd": Path(cfg.usd.output_dir or (cfg.workspace_dir / "usd")).resolve(),
        "ns_data": Path(cfg.nerfstudio_data_dir).resolve(),
        "renders": Path(cfg.renders_dir).resolve(),
        "dataset": Path(cfg.dataset_dir).resolve(),
    }


def _resolve_under(root: Path, rel: str) -> Path:
    target = (root / rel).resolve() if rel else root
    if not str(target).startswith(str(root)):
        raise HTTPException(403, "Path escapes allowlisted root")
    return target


@router.get("/roots")
def list_roots() -> dict:
    try:
        cfg = _cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    roots = _roots(cfg)
    return {
        "roots": [
            {"id": k, "path": str(v), "exists": v.exists()}
            for k, v in roots.items()
        ]
    }


@router.get("/list")
def list_dir(root: str = Query(...), path: str = Query("")) -> dict:
    try:
        cfg = _cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    roots = _roots(cfg)
    if root not in roots:
        raise HTTPException(400, f"Unknown root: {root}")
    base = roots[root]
    target = _resolve_under(base, path)
    if not target.exists():
        return {"path": path, "entries": [], "exists": False}
    if target.is_file():
        raise HTTPException(400, "Not a directory")
    entries = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    for child in children[:2000]:
        entries.append(
            {
                "name": child.name,
                "path": str(child.relative_to(base)),
                "is_dir": child.is_dir(),
                "size": child.stat().st_size if child.is_file() else None,
                "suffix": child.suffix.lower(),
            }
        )
    return {"root": root, "path": path, "abs": str(target), "entries": entries, "exists": True}


@router.get("/file")
def get_file(root: str = Query(...), path: str = Query(...)) -> FileResponse:
    try:
        cfg = _cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    roots = _roots(cfg)
    if root not in roots:
        raise HTTPException(400, f"Unknown root: {root}")
    target = _resolve_under(roots[root], path)
    if not target.is_file():
        raise HTTPException(404, "File not found")
    media, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media or "application/octet-stream")
