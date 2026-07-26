"""Pipeline state + manifest summary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import SceneManifest

from scan2usd_gui.schema import LEGACY_STAGES, PIPELINE_STAGES
from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def _load_cfg() -> SceneConfig:
    path = project_state.require_config()
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        return SceneConfig.load(path)
    finally:
        os.chdir(prev)


@router.get("/state")
def pipeline_state() -> dict[str, Any]:
    try:
        cfg = _load_cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc

    state_path = cfg.workspace_dir / "build" / "pipeline_state.json"
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text())

    manifest_summary: dict[str, Any] | None = None
    objects: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = SceneManifest.load(manifest_path)
        try:
            approvals = json.loads(
                json.dumps(dict(getattr(manifest, "approvals", {}) or {}), default=str)
            )
        except Exception:  # noqa: BLE001
            approvals = {}

        objects = [
            json.loads(
                json.dumps(
                    {
                        "instance_id": o.instance_id,
                        "class_name": o.class_name,
                        "movable": o.movable,
                        "review_state": o.review_state,
                        "observed_background_coverage": o.observed_background_coverage,
                        "physics": o.physics,
                    },
                    default=str,
                )
            )
            for o in manifest.objects
        ]
        try:
            artifacts = json.loads(json.dumps(getattr(manifest, "artifacts", []), default=str))
        except Exception:  # noqa: BLE001
            artifacts = []
        manifest_summary = {
            "scene_name": manifest.scene_name,
            "build_mode": manifest.build_mode,
            "review_state": getattr(manifest, "review_state", None),
            "approvals": approvals,
            "object_count": len(objects),
            "artifacts": artifacts,
        }

    stages_done: set[str] = set()
    raw_stages = state.get("stages")
    if isinstance(raw_stages, dict):
        for name, info in raw_stages.items():
            if isinstance(info, dict):
                status = str(info.get("status", "")).lower()
                if status in {"done", "completed", "ok", "success"}:
                    stages_done.add(str(name))
            elif isinstance(info, str) and info.lower() in {"done", "completed", "ok", "success"}:
                stages_done.add(str(name))
            elif info is True:
                stages_done.add(str(name))
    completed = state.get("completed")
    if isinstance(completed, list):
        stages_done.update(str(x) for x in completed)

    # Map orchestrator stage names → Pipeline tile ids / CLI commands.
    if "mask_propagation" in stages_done or (
        "object_proposals" in stages_done and "mask_propagation" in stages_done
    ):
        stages_done.add("segment-usd")
    if "floor_alignment" in stages_done:
        stages_done.add("align-floor")
    if "usd_package" in stages_done:
        stages_done.add("build-usd")

    # Infer tile completion from workspace / manifest (CLI steps often skip pipeline_state).
    ns_data = Path(cfg.nerfstudio_data_dir) if cfg.nerfstudio_data_dir else None
    if ns_data and (
        (ns_data / "colmap" / "sparse" / "0").is_dir()
        or (ns_data / "sparse_pc.ply").is_file()
        or (ns_data / "transforms.json").is_file()
    ):
        stages_done.add("reconstruct")
    if manifest_path.is_file():
        stages_done.add("init-usd")

    masks_root = Path(
        cfg.segmentation.masks_dir or (cfg.workspace_dir / "masks")
    )
    if masks_root.is_dir() and any(
        p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        for p in masks_root.rglob("*")
    ):
        stages_done.add("segment-usd")

    floor_json = cfg.workspace_dir / "colmap_to_usd_floor.json"
    if floor_json.is_file() or "floor_alignment" in {
        str(a.get("artifact_id"))
        for a in (manifest_summary or {}).get("artifacts") or []
        if isinstance(a, dict)
    }:
        stages_done.add("align-floor")

    approvals = (manifest_summary or {}).get("approvals") or {}
    if any(
        isinstance(o, dict) and str(o.get("review_state", "")).lower() == "approved"
        for o in objects
    ):
        stages_done.add("review")
    if isinstance(approvals, dict):
        metric_gate = approvals.get("metric_transform") or {}
        metric_json = cfg.workspace_dir / "colmap_to_usd_metric.json"
        scale_method = ""
        if manifest_path.is_file():
            try:
                scale_method = str(
                    getattr(SceneManifest.load(manifest_path).scale, "method", "") or ""
                ).lower()
            except Exception:  # noqa: BLE001
                scale_method = ""
        measured = any(
            token in scale_method
            for token in ("measured", "meters_per_unit", "calibrated", "metric")
        )
        if metric_json.is_file() or (
            isinstance(metric_gate, dict)
            and str(metric_gate.get("state", "")).lower() == "approved"
            and measured
        ):
            stages_done.add("apply-metric-scale")

    root_usd = None
    for a in (manifest_summary or {}).get("artifacts") or []:
        if isinstance(a, dict) and a.get("artifact_id") == "root_usd":
            root_usd = a.get("path")
            break
    if root_usd and Path(str(root_usd)).is_file():
        stages_done.add("build-usd")
    usd_dir = Path(cfg.usd.output_dir or (cfg.workspace_dir / "usd"))
    if (usd_dir / "build_report.json").is_file():
        stages_done.add("validate-usd")

    return {
        "pipeline_stages": PIPELINE_STAGES,
        "legacy_stages": LEGACY_STAGES,
        "pipeline_state": state,
        "stages_done": sorted(stages_done),
        "manifest": manifest_summary,
        "objects": objects,
        "paths": {
            "state": str(state_path),
            "manifest": str(manifest_path),
            "workspace": str(cfg.workspace_dir),
        },
    }
