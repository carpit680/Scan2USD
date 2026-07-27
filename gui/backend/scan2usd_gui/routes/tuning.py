"""Auto-tuner trial data for the Tuning page."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from scan2usd.config import SceneConfig

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/tuning", tags=["tuning"])


def _load_cfg() -> SceneConfig:
    path = project_state.require_config()
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        return SceneConfig.load(path)
    finally:
        os.chdir(prev)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@router.get("/trials")
def tuning_trials() -> dict[str, Any]:
    """Trial history + current scene quality for the Tuning page (poll-friendly)."""
    try:
        cfg = _load_cfg()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc

    trials_raw = _read_json(cfg.workspace_dir / "tuning" / "trials.json") or {}
    trials = list(trials_raw.get("trials", []))
    best = None
    scored = [t for t in trials if t.get("status") == "scored" and t.get("quality_score") is not None]
    if scored:
        best = max(scored, key=lambda t: t["quality_score"])

    usd_dir = Path(cfg.usd.output_dir or cfg.workspace_dir / "usd")
    quality = _read_json(usd_dir / "scene_quality.json")
    raw_splat = cfg.workspace_dir / "build" / "visual" / "environment_splat_raw.usd"
    heldout = cfg.workspace_dir / "build" / "grut_dataset" / "held_out.json"

    tuned_config: str | None = None
    config_path = project_state.config_path
    if config_path:
        candidate = Path(config_path).with_name(f"{Path(config_path).stem}_tuned.yaml")
        if candidate.is_file():
            tuned_config = str(candidate.resolve())

    return {
        "trials": trials,
        "best_trial": best,
        "scene_quality": quality,
        "tuned_config": tuned_config,
        "budgets": {
            "max_cheap_trials": cfg.tuning.max_cheap_trials,
            "max_retrain_trials": cfg.tuning.max_retrain_trials,
            "lpips": cfg.tuning.lpips,
        },
        "ready": {
            "raw_splat": raw_splat.is_file(),
            "held_out": heldout.is_file(),
            "isaac_python": bool(str(cfg.external.get("isaac_python", "")).strip()),
        },
        "paths": {
            "trials_json": str((cfg.workspace_dir / "tuning" / "trials.json").resolve()),
            "scene_quality": str((usd_dir / "scene_quality.json").resolve()),
        },
    }
