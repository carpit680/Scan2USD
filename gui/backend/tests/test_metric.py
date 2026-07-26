"""Metric scene endpoint tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.state import project_state


def test_metric_scene_missing_files(tmp_path: Path):
    cfg = tmp_path / "scene.yaml"
    ws = tmp_path / "workspace"
    ns = tmp_path / "ns_data"
    ws.mkdir()
    ns.mkdir()
    cfg.write_text(
        "\n".join(
            [
                "name: test",
                f"workspace_dir: {ws}",
                f"nerfstudio_data_dir: {ns}",
                "capture:",
                "  modality: rgb",
            ]
        )
    )
    project_state.set_project(cfg, cwd=tmp_path)
    client = TestClient(create_app())
    r = client.get("/api/metric/scene")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_ply"] is False
    assert body["has_floor"] is False
    assert body["ready"] is False
    assert body["ply_url"] is None
    assert body["floor_matrix"] is None


def test_metric_scene_with_ply_and_floor(tmp_path: Path):
    cfg = tmp_path / "scene.yaml"
    ws = tmp_path / "workspace"
    ns = tmp_path / "ns_data"
    ws.mkdir()
    ns.mkdir()
    (ns / "sparse_pc.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n0 0 0\n"
    )
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    (ws / "colmap_to_usd_floor.json").write_text(
        json.dumps({"colmap_to_usd": identity, "method": "test"}) + "\n"
    )
    cfg.write_text(
        "\n".join(
            [
                "name: test",
                f"workspace_dir: {ws}",
                f"nerfstudio_data_dir: {ns}",
                "capture:",
                "  modality: rgb",
            ]
        )
    )
    project_state.set_project(cfg, cwd=tmp_path)
    client = TestClient(create_app())
    r = client.get("/api/metric/scene")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_ply"] is True
    assert body["has_floor"] is True
    assert body["ready"] is True
    assert body["ply_url"] == "/api/artifacts/file?root=ns_data&path=sparse_pc.ply"
    assert body["floor_matrix"] == identity
    assert body["meters_approved"] is False
