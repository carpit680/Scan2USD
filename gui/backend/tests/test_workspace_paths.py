"""Workspace path defaults for Commands / tools."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from scan2usd.config import SceneConfig
from scan2usd_gui.app import create_app
from scan2usd_gui.bridge import apply_command_path_defaults, workspace_paths, workspace_summary
from scan2usd_gui.state import project_state


def test_workspace_paths_follow_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace_desk" / "usd").mkdir(parents=True)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "desk",
                "workspace_dir": "workspace_desk",
                "usd": {"root_filename": "desk.usd"},
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    paths = workspace_paths(cfg, cwd=tmp_path)
    assert paths["workspace_dir"] == "workspace_desk"
    assert paths["root_usd"] == "workspace_desk/usd/desk.usd"
    assert paths["environment_splat"] == "workspace_desk/build/visual/environment_splat.usd"
    assert paths["frames_dir"] == "workspace_desk/frames"

    summary = workspace_summary(cfg, cwd=tmp_path)
    assert summary["paths"]["root_usd"] == "workspace_desk/usd/desk.usd"


def test_apply_command_path_defaults_isaac_view(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "desk",
                "workspace_dir": "workspace_desk",
                "usd": {"root_filename": "desk.usd"},
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    filled = apply_command_path_defaults("tool-isaac-view", {}, cfg, cwd=tmp_path)
    assert filled["stage"] == "workspace_desk/usd/desk.usd"


def test_apply_cleanup_params_from_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "desk",
                "workspace_dir": "workspace_desk",
                "reconstruction": {
                    "splat_cleanup": {"outlier_std": 2.0, "min_opacity": 0.05},
                },
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    hybrid = apply_command_path_defaults("cleanup-splat", {}, cfg, cwd=tmp_path)
    assert hybrid["outlier_std"] == 2.0
    assert hybrid["min_opacity"] == 0.05
    tool = apply_command_path_defaults("tool-cleanup-splat-usd", {}, cfg, cwd=tmp_path)
    assert tool["outlier_std"] == 2.0
    assert tool["min_opacity"] == 0.05


def test_project_api_exposes_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "desk",
                "workspace_dir": "workspace_desk",
                "usd": {"root_filename": "desk.usd"},
            }
        )
    )
    app = create_app()
    client = TestClient(app)
    project_state.set_project(cfg_path, cwd=tmp_path)
    resp = client.put("/api/project", json={"config_path": str(cfg_path), "cwd": str(tmp_path)})
    assert resp.status_code == 200, resp.text
    paths = resp.json()["workspace"]["paths"]
    assert paths["root_usd"] == "workspace_desk/usd/desk.usd"
    assert paths["colmap_to_usd_floor"] == "workspace_desk/colmap_to_usd_floor.json"
