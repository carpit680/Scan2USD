"""Review API wrapping ReviewSession."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from scan2usd.config import SceneConfig
from scan2usd.review.app import ReviewSession

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/review", tags=["review"])


def _session() -> ReviewSession:
    path = project_state.require_config()
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        cfg = SceneConfig.load(path)
    finally:
        os.chdir(prev)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run init-usd / segment-usd first."
        )
    return ReviewSession(cfg, manifest_path)


@router.get("/instances")
def list_instances() -> dict[str, Any]:
    try:
        session = _session()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    items = []
    for iid in session.instance_ids():
        obj = session.object(iid)
        mask_count = len(session.mask_files(iid))
        items.append(
            {
                "instance_id": obj.instance_id,
                "class_name": obj.class_name,
                "movable": obj.movable,
                "review_state": obj.review_state,
                "observed_background_coverage": obj.observed_background_coverage,
                "physics_template": obj.physics.get("template", "generic"),
                "notes": obj.physics.get("review_notes", ""),
                "mask_count": mask_count,
                "merged_into": obj.physics.get("merged_into"),
                "merged_from": obj.physics.get("merged_from") or [],
            }
        )
    masked = [
        item
        for item in items
        if int(item["mask_count"]) > 0 and not item.get("merged_into")
    ]
    return {
        "instances": masked,
        "total_instances": len(items),
        "masked_instances": len(masked),
        "merged_away": sum(1 for item in items if item.get("merged_into")),
        "approvals": _approvals_dict(session),
        "build_mode": session.manifest.build_mode,
        "scene_name": session.manifest.scene_name,
    }


def _approvals_dict(session: ReviewSession) -> dict[str, Any]:
    import json

    try:
        raw = getattr(session.manifest, "approvals", {}) or {}
        return json.loads(json.dumps(dict(raw), default=str))
    except Exception:  # noqa: BLE001
        return {}


@router.get("/instances/{instance_id}")
def get_instance(instance_id: str) -> dict[str, Any]:
    try:
        session = _session()
        obj = session.object(instance_id)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc
    overlays = session.render_overlays(instance_id)
    mask_files = session.mask_files(instance_id)
    return {
        "instance_id": obj.instance_id,
        "class_name": obj.class_name,
        "movable": obj.movable,
        "review_state": obj.review_state,
        "observed_background_coverage": obj.observed_background_coverage,
        "physics_template": str(obj.physics.get("template", "generic")),
        "notes": str(obj.physics.get("review_notes", "")),
        "physics": obj.physics,
        "merged_into": obj.physics.get("merged_into"),
        "merged_from": obj.physics.get("merged_from") or [],
        "mask_count": len(mask_files),
        "mask_files": [str(p) for p in mask_files],
        "overlays": overlays,
        "render_mesh": obj.render_mesh,
        "collision_mesh": obj.collision_mesh,
        "baked_texture": obj.baked_texture,
    }


class UpdateInstanceBody(BaseModel):
    class_name: str
    movable: bool
    review_state: str
    observed_background_coverage: float = Field(ge=0, le=1)
    physics_template: str = "generic"
    notes: str = ""
    reclassify: bool = True


@router.get("/instances/{instance_id}/reclassify-preview")
def reclassify_preview(instance_id: str, class_name: str) -> dict[str, Any]:
    try:
        session = _session()
        return session.preview_reclassify(instance_id, class_name)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc


@router.put("/instances/{instance_id}")
def update_instance(instance_id: str, body: UpdateInstanceBody) -> dict[str, Any]:
    try:
        session = _session()
        preview = session.preview_reclassify(instance_id, body.class_name)
        obj = session.update_object(
            instance_id,
            class_name=body.class_name,
            movable=body.movable,
            review_state=body.review_state,
            observed_background_coverage=body.observed_background_coverage,
            physics_template=body.physics_template,
            notes=body.notes,
            reclassify=body.reclassify,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return {
        "instance_id": obj.instance_id,
        "previous_instance_id": instance_id,
        "renamed": obj.instance_id != instance_id,
        "class_name": obj.class_name,
        "movable": obj.movable,
        "review_state": obj.review_state,
        "observed_background_coverage": obj.observed_background_coverage,
        "will_rename": preview["will_rename"],
        "preview_new_instance_id": preview["new_instance_id"],
    }


@router.post("/instances/{instance_id}/masks")
async def upload_masks(instance_id: str, files: list[UploadFile] = File(...)) -> dict:
    try:
        session = _session()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    saved: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        for uf in files:
            dest = Path(tmp) / (uf.filename or "mask.png")
            dest.write_bytes(await uf.read())
            saved.append(dest)
        count = session.import_corrected_masks(instance_id, saved)
    return {"imported": count}


@router.delete("/instances/{instance_id}/masks/{mask_name}")
def delete_mask(instance_id: str, mask_name: str) -> dict[str, Any]:
    try:
        session = _session()
        result = session.delete_mask(instance_id, mask_name)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    # Preserve single-delete shape used by older clients
    deleted = result.get("deleted") or []
    return {
        "ok": True,
        "instance_id": result["instance_id"],
        "deleted": deleted[0] if len(deleted) == 1 else deleted,
        "mask_count": result["mask_count"],
        "missing": result.get("missing") or [],
    }


class DeleteMasksBody(BaseModel):
    mask_names: list[str] = Field(default_factory=list)


@router.post("/instances/{instance_id}/masks/delete")
def delete_masks(instance_id: str, body: DeleteMasksBody) -> dict[str, Any]:
    if not body.mask_names:
        raise HTTPException(400, "Provide one or more mask_names to delete")
    try:
        session = _session()
        result = session.delete_masks(instance_id, body.mask_names)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


class MergeBody(BaseModel):
    source_ids: list[str] = Field(default_factory=list)


@router.post("/instances/{instance_id}/merge")
def merge_instances(instance_id: str, body: MergeBody) -> dict[str, Any]:
    try:
        session = _session()
        result = session.merge_instances(instance_id, body.source_ids)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


class UnmergeBody(BaseModel):
    source_ids: list[str] | None = None


@router.post("/instances/{instance_id}/unmerge")
def unmerge_instances(instance_id: str, body: UnmergeBody | None = None) -> dict[str, Any]:
    try:
        session = _session()
        source_ids = None if body is None else body.source_ids
        result = session.unmerge_instances(instance_id, source_ids)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, **result}


@router.get("/overlay")
def get_overlay(path: str) -> FileResponse:
    """Serve an overlay image under the workspace (path allowlist)."""
    try:
        cfg_path = project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        cfg = SceneConfig.load(cfg_path)
    finally:
        os.chdir(prev)
    target = Path(path).resolve()
    root = Path(cfg.workspace_dir).resolve()
    if not str(target).startswith(str(root)) or not target.is_file():
        raise HTTPException(403, "Path not allowed")
    return FileResponse(target)
