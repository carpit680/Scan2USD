"""Tuning trials endpoint tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.state import project_state


def _project(tmp_path: Path) -> Path:
    cfg = tmp_path / "scene.yaml"
    ws = tmp_path / "workspace"
    ws.mkdir()
    cfg.write_text(f"name: test\nworkspace_dir: {ws}\n")
    project_state.set_project(cfg, cwd=tmp_path)
    return ws


def test_trials_empty_workspace(tmp_path: Path):
    _project(tmp_path)
    client = TestClient(create_app())
    response = client.get("/api/tuning/trials")
    assert response.status_code == 200
    data = response.json()
    assert data["trials"] == []
    assert data["best_trial"] is None
    assert data["ready"]["raw_splat"] is False
    assert data["budgets"]["max_cheap_trials"] == 12


def test_trials_with_history_and_quality(tmp_path: Path):
    ws = _project(tmp_path)
    tuning_dir = ws / "tuning"
    tuning_dir.mkdir()
    (tuning_dir / "trials.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "trials": [
                    {
                        "trial_id": "cheap_001",
                        "kind": "cheap",
                        "params": {"outlier_std": 2.0},
                        "status": "scored",
                        "quality_score": 55.0,
                    },
                    {
                        "trial_id": "cheap_002",
                        "kind": "cheap",
                        "params": {"outlier_std": 4.0},
                        "status": "scored",
                        "quality_score": 71.5,
                    },
                    {
                        "trial_id": "cheap_003",
                        "kind": "cheap",
                        "params": {"outlier_std": 6.0},
                        "status": "failed",
                        "quality_score": None,
                    },
                ],
            }
        )
    )
    usd = ws / "usd"
    usd.mkdir()
    (usd / "scene_quality.json").write_text(json.dumps({"quality_score": 71.5}))

    client = TestClient(create_app())
    data = client.get("/api/tuning/trials").json()
    assert len(data["trials"]) == 3
    assert data["best_trial"]["trial_id"] == "cheap_002"
    assert data["scene_quality"]["quality_score"] == 71.5
