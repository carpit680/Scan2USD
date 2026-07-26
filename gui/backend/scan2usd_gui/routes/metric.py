"""Metric scale scene data for the in-app edge picker."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from scan2usd.config import SceneConfig
from scan2usd.geometry.metric_scale import resolve_floor_transform_path
from scan2usd.pipeline.manifest import SceneManifest

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/metric", tags=["metric"])


def _load_cfg() -> SceneConfig:
    path = project_state.require_config()
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        return SceneConfig.load(path)
    finally:
        os.chdir(prev)


@router.get("/scene")
def metric_scene() -> dict[str, Any]:
    try:
        cfg = _load_cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc

    ns_data = Path(cfg.nerfstudio_data_dir) if cfg.nerfstudio_data_dir else None
    ply_path = (ns_data / "sparse_pc.ply") if ns_data else None
    has_ply = bool(ply_path and ply_path.is_file())

    floor_matrix: list[list[float]] | None = None
    floor_path: str | None = None
    has_floor = False
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    transforms: list[Any] = []
    meters_approved = False
    existing_scale: float | None = None
    scale_method: str | None = None

    if manifest_path.is_file():
        manifest = SceneManifest.load(manifest_path)
        transforms = list(manifest.transforms or [])
        scale = manifest.scale
        meters_approved = bool(getattr(scale, "approved", False))
        existing_scale = (
            float(scale.meters_per_source_unit)
            if scale.meters_per_source_unit is not None
            else None
        )
        scale_method = str(scale.method) if scale.method else None
        approvals = getattr(manifest, "approvals", {}) or {}
        gate = approvals.get("metric_transform") or {}
        if gate.get("state") == "approved":
            meters_approved = True

    try:
        resolved = resolve_floor_transform_path(cfg.workspace_dir, transforms)
        if resolved.is_file():
            has_floor = True
            floor_path = str(resolved.resolve())
            raw = json.loads(resolved.read_text(encoding="utf-8"))
            matrix = raw.get("colmap_to_usd") if isinstance(raw, dict) else raw
            if isinstance(matrix, list) and len(matrix) == 4:
                floor_matrix = matrix
    except FileNotFoundError:
        preferred = cfg.workspace_dir / "colmap_to_usd_floor.json"
        if preferred.is_file():
            has_floor = True
            floor_path = str(preferred.resolve())
            raw = json.loads(preferred.read_text(encoding="utf-8"))
            matrix = raw.get("colmap_to_usd") if isinstance(raw, dict) else raw
            if isinstance(matrix, list) and len(matrix) == 4:
                floor_matrix = matrix

    ply_url = (
        "/api/artifacts/file?root=ns_data&path=sparse_pc.ply" if has_ply else None
    )

    return {
        "has_ply": has_ply,
        "has_floor": has_floor and floor_matrix is not None,
        "ply_path": str(ply_path.resolve()) if has_ply and ply_path else None,
        "ply_url": ply_url,
        "floor_path": floor_path,
        "floor_matrix": floor_matrix,
        "meters_approved": meters_approved,
        "existing_scale": existing_scale,
        "scale_method": scale_method,
        "ready": has_ply and has_floor and floor_matrix is not None,
    }
