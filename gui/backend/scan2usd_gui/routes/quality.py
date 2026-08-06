"""One place to see whether the scene is any good.

The numbers that matter live in four different JSON files written by four
different stages, and until now reading them meant knowing which. Worse, two of
them answer different questions and are easy to confuse: held-out PSNR says
whether the training views reproduce, and is structurally blind to interior
haze, because haze between the camera and a wall renders roughly the pixels the
wall would. The fog metrics are the ones that see that.

This endpoint gathers them and says which is which.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from scan2usd.config import SceneConfig

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/quality", tags=["quality"])


def _cfg() -> SceneConfig:
    path = project_state.require_config()
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        return SceneConfig.load(path)
    finally:
        os.chdir(prev)


def _load(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"value": data}
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _grade(value: float | None, good: float, poor: float) -> str:
    """pass / warn / fail against a scale that may run in either direction."""
    if value is None:
        return "unknown"
    if good >= poor:
        return "pass" if value >= good else ("warn" if value >= poor else "fail")
    return "pass" if value <= good else ("warn" if value <= poor else "fail")


@router.get("")
def get_quality() -> dict:
    try:
        cfg = _cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

    workspace = Path(cfg.workspace_dir)
    build = workspace / "build"
    splat_dir = build / "visual"
    usd_dir = Path(cfg.usd.output_dir or (workspace / "usd"))

    cleanup = _load(splat_dir / "splat_cleanup_report.json")
    analysis = _load(splat_dir / "splat_analysis.json")
    scene_quality = _load(build / "scene_quality.json") or _load(usd_dir / "scene_quality.json")
    build_report = _load(usd_dir / "build_report.json")
    gate_report = _load(build / "quality_gates.json")
    baseline = _load(splat_dir / "baseline" / "scene_quality.json")

    fog = (cleanup or {}).get("fog_metrics") or {}
    photo = (scene_quality or {}).get("photorealism") or scene_quality or {}

    def metric(
        key: str,
        label: str,
        value: Any,
        *,
        good: float | None = None,
        poor: float | None = None,
        unit: str = "",
        note: str = "",
    ) -> dict:
        numeric = value if isinstance(value, (int, float)) else None
        return {
            "key": key,
            "label": label,
            "value": numeric,
            "unit": unit,
            "status": _grade(numeric, good, poor) if good is not None else "info",
            "note": note,
        }

    appearance = [
        metric(
            "quality_score",
            "Quality score",
            (scene_quality or {}).get("quality_score"),
            good=70,
            poor=50,
            note="Weighted SSIM/LPIPS/PSNR against held-out capture frames, minus gate penalties.",
        ),
        metric("psnr", "PSNR", photo.get("mean_psnr"), good=22, poor=18, unit="dB"),
        metric("ssim", "SSIM", photo.get("mean_ssim"), good=0.8, poor=0.65),
        metric("lpips", "LPIPS", photo.get("mean_lpips"), good=0.3, poor=0.45,
               note="Perceptual distance; lower is better."),
        metric("views", "Views scored", photo.get("evaluated_views")),
    ]

    clarity = [
        metric(
            "transmittance",
            "Seen across the room",
            fog.get("transmittance_across_room"),
            good=0.8,
            poor=0.4,
            unit="fraction",
            note=(
                "Fraction of a view that survives crossing the room. This is what "
                "'cloudy' means, and held-out PSNR cannot see it."
            ),
        ),
        metric(
            "fog_inside",
            "Haze in the walked volume",
            fog.get("fog_inside_hull"),
            note="Gaussians inside the capture path with no surface near them.",
        ),
        metric(
            "fog_fraction",
            "Haze share of that volume",
            fog.get("fog_fraction_of_inside"),
            good=0.05,
            poor=0.2,
            unit="fraction",
        ),
        metric("mean_free_path", "Mean free path", fog.get("mean_free_path"), unit="units"),
    ]

    removed = {
        key.replace("removed_", ""): cleanup.get(key)
        for key in (cleanup or {})
        if key.startswith("removed_") and cleanup.get(key)
    }

    warnings: list[str] = []
    manifest = _load(workspace / "scene_manifest.json") or {}
    for item in manifest.get("warnings", []) or []:
        if item not in warnings:
            warnings.append(str(item))
    if (cleanup or {}).get("free_space_error"):
        warnings.append(f"Free-space carving: {cleanup['free_space_error']}")
    scale = manifest.get("scale") or {}
    if scale.get("meters_per_source_unit") in (1.0, 1, None) and "unit_scale" in str(
        scale.get("method", "")
    ):
        warnings.append(
            "Scene is not metric: distances are in COLMAP units, so physics and "
            "any RL policy trained on it will have the wrong scale."
        )

    return {
        "workspace": str(workspace),
        "appearance": appearance,
        "clarity": clarity,
        "removed": removed,
        "kept": (cleanup or {}).get("kept_count"),
        "input_count": (cleanup or {}).get("input_count"),
        "population": (analysis or {}).get("population"),
        "blocking": (analysis or {}).get("blocking_cross_section"),
        "baseline_score": (baseline or {}).get("quality_score"),
        "gates": (build_report or {}).get("checks"),
        "quality_gates": (gate_report or {}).get("gates") or [],
        "usable": (build_report or {}).get("usable"),
        "warnings": warnings,
        "have": {
            "cleanup": cleanup is not None,
            "analysis": analysis is not None,
            "scene_quality": scene_quality is not None,
            "build_report": build_report is not None,
            "quality_gates": gate_report is not None,
        },
    }


@router.get("/preview.ply")
def preview_ply(small: bool = False):
    """
    Serve the browser-preview PLY, streamed so a 66 MB splat does not buffer.

    Range requests matter here: the viewer asks for the header before the body,
    and without them the whole file is refetched.
    """
    from fastapi.responses import FileResponse

    try:
        cfg = _cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    name = "preview_small.ply" if small else "preview.ply"
    path = Path(cfg.workspace_dir) / "build" / "visual" / name
    if not path.is_file():
        raise HTTPException(
            404,
            "No preview.ply yet — run the 'Export splat PLY' tool to build one "
            "from the cleaned splat.",
        )
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/preview-status")
def preview_status() -> dict:
    try:
        cfg = _cfg()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    path = Path(cfg.workspace_dir) / "build" / "visual" / "preview.ply"
    source = Path(cfg.workspace_dir) / "build" / "visual" / "environment_splat.usd"
    small = Path(cfg.workspace_dir) / "build" / "visual" / "preview_small.ply"
    meta = _load(Path(cfg.workspace_dir) / "build" / "visual" / "preview_meta.json")
    exists = path.is_file()
    return {
        "meta": meta,
        "has_small": small.is_file(),
        "small_bytes": small.stat().st_size if small.is_file() else 0,
        "exists": exists,
        "path": str(path),
        "bytes": path.stat().st_size if exists else 0,
        # A preview older than the splat it came from is worse than none: it
        # shows a cleanup setting you already changed.
        "stale": bool(
            exists and source.is_file() and source.stat().st_mtime > path.stat().st_mtime
        ),
    }
