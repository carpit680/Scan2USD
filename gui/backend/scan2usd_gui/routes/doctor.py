"""Doctor endpoint."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from scan2usd.config import SceneConfig
from scan2usd.doctor_deps import collect_doctor_results

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/doctor", tags=["doctor"])


@router.get("")
def doctor() -> dict:
    try:
        path = project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        cfg = SceneConfig.load(path)
        report = collect_doctor_results(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    finally:
        os.chdir(prev)

    groups = {
        name: [
            {
                "label": it.label,
                "ok": it.ok,
                "detail": it.detail,
                "required": it.required,
                "apt_if_missing": list(it.apt_if_missing),
                "pip_hint": it.pip_hint,
            }
            for it in items
        ]
        for name, items in report.groups.items()
    }
    return {
        "groups": groups,
        "apt_packages": list(report.apt_packages),
        "apt_install_line": report.apt_install_line,
        "reconstruct_ready": report.reconstruct_ready,
    }
