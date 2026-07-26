"""Config save syncs workspace-derived paths."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.state import project_state


def test_put_config_strips_paths_when_workspace_changes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "workspace_dir": "workspace_a",
                "frames_dir": "workspace_a/frames",
                "segmentation": {"masks_dir": "workspace_a/masks", "mask_model": "sam2"},
            }
        )
    )
    app = create_app()
    client = TestClient(app)
    project_state.set_project(cfg_path, cwd=tmp_path)

    resp = client.put(
        "/api/config",
        json={"raw": {"name": "t", "workspace_dir": "workspace_b", "frames_dir": "workspace_a/frames"}},
    )
    assert resp.status_code == 200, resp.text
    saved = yaml.safe_load(cfg_path.read_text())
    assert saved["workspace_dir"] == "workspace_b"
    assert "frames_dir" not in saved

    # Resolved config follows new workspace
    cfg = resp.json()["config"]
    assert cfg["frames_dir"].endswith("workspace_b/frames")
    assert cfg["segmentation"]["masks_dir"].endswith("workspace_b/masks")
