"""Path resolution in SceneConfig.load."""

from __future__ import annotations

from pathlib import Path

import yaml

from scan2usd.config import SceneConfig


def test_relative_paths_resolve_to_cwd(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "sub" / "scene.yaml"
    cfg_path.parent.mkdir(parents=True)
    (tmp_path / "workspace" / "frames").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "frames_dir": "workspace/frames",
                "workspace_dir": "workspace",
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    assert cfg.frames_dir == tmp_path / "workspace" / "frames"
    assert cfg.workspace_dir == tmp_path / "workspace"


def test_paths_relative_to_config(tmp_path: Path, monkeypatch) -> None:
    """Anchor relative paths next to the YAML file."""
    sub = tmp_path / "configs"
    sub.mkdir()
    (sub / "workspace" / "frames").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    cfg_path = sub / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths_relative_to": "config",
                "frames_dir": "workspace/frames",
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    assert cfg.frames_dir == sub / "workspace" / "frames"


def test_external_tool_names_are_strings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "frames_dir": "workspace/frames",
                "external": {"colmap": "colmap", "ns_process_data": "ns-process-data"},
            }
        )
    )
    (tmp_path / "workspace" / "frames").mkdir(parents=True)
    cfg = SceneConfig.load(cfg_path)
    assert cfg.external["ns_process_data"] == "ns-process-data"
    assert isinstance(cfg.external["ns_process_data"], str)
