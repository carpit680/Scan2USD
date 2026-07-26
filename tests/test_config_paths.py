"""Path resolution in SceneConfig.load."""

from __future__ import annotations

from pathlib import Path

import yaml

from scan2usd.config import (
    SceneConfig,
    strip_workspace_derived_paths,
    sync_workspace_paths,
)


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


def test_workspace_dir_derives_layout_paths(tmp_path: Path, monkeypatch) -> None:
    """Only workspace_dir is required — frames/masks/usd follow automatically."""
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "desk", "workspace_dir": "workspace_desk"}))
    cfg = SceneConfig.load(cfg_path)
    ws = tmp_path / "workspace_desk"
    assert cfg.workspace_dir == ws
    assert cfg.frames_dir == ws / "frames"
    assert cfg.colmap_txt_dir == ws / "colmap_txt"
    assert cfg.nerfstudio_data_dir == ws / "ns_data"
    assert cfg.renders_dir == ws / "renders"
    assert cfg.dataset_dir == ws / "dataset"
    assert cfg.segmentation.masks_dir == ws / "masks"
    assert cfg.usd.output_dir == ws / "usd"


def test_explicit_path_override_preserved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "elsewhere" / "frames"
    other.mkdir(parents=True)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": "workspace_desk",
                "frames_dir": str(other),
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    assert cfg.frames_dir == other.resolve()
    assert cfg.nerfstudio_data_dir == tmp_path / "workspace_desk" / "ns_data"


def test_sync_strips_derived_when_workspace_changes() -> None:
    raw = {
        "workspace_dir": "workspace_b",
        "frames_dir": "workspace_a/frames",
        "segmentation": {"masks_dir": "workspace_a/masks", "mask_model": "sam2"},
    }
    out = sync_workspace_paths(raw, previous_workspace_dir="workspace_a")
    assert out["workspace_dir"] == "workspace_b"
    assert "frames_dir" not in out
    assert "masks_dir" not in out.get("segmentation", {})
    assert out["segmentation"]["mask_model"] == "sam2"


def test_sync_drops_redundant_defaults() -> None:
    raw = {
        "workspace_dir": "workspace_desk",
        "frames_dir": "workspace_desk/frames",
        "usd": {"output_dir": "workspace_desk/usd", "root_filename": "desk.usd"},
    }
    out = sync_workspace_paths(raw)
    assert "frames_dir" not in out
    assert "output_dir" not in out.get("usd", {})
    assert out["usd"]["root_filename"] == "desk.usd"


def test_strip_workspace_derived_paths() -> None:
    raw = {
        "workspace_dir": "ws",
        "frames_dir": "/abs/frames",
        "colmap_txt_dir": "ws/colmap_txt",
    }
    out = strip_workspace_derived_paths(raw)
    assert out["workspace_dir"] == "ws"
    assert "frames_dir" not in out
    assert "colmap_txt_dir" not in out
