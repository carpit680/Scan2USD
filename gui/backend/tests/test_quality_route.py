"""The quality endpoint must degrade to 'not measured yet', never to a wrong number."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.state import project_state


def _project(tmp_path: Path, *, cleanup: dict | None = None, quality: dict | None = None):
    workspace = tmp_path / "ws"
    (workspace / "build" / "visual").mkdir(parents=True)
    if cleanup is not None:
        (workspace / "build" / "visual" / "splat_cleanup_report.json").write_text(
            json.dumps(cleanup)
        )
    if quality is not None:
        (workspace / "build" / "scene_quality.json").write_text(json.dumps(quality))
    config = tmp_path / "scene.yaml"
    config.write_text(yaml.safe_dump({"name": "t", "workspace_dir": str(workspace)}))
    project_state.set_project(config, cwd=tmp_path)
    return TestClient(create_app())


def test_missing_reports_report_absence_rather_than_zero(tmp_path):
    body = _project(tmp_path).get("/api/quality").json()
    assert body["have"] == {
        "cleanup": False,
        "analysis": False,
        "scene_quality": False,
        "build_report": False,
    }
    for metric in body["appearance"] + body["clarity"]:
        assert metric["value"] is None
        assert metric["status"] in {"info", "unknown"}


def test_fog_and_appearance_are_graded_in_their_own_directions(tmp_path):
    client = _project(
        tmp_path,
        cleanup={
            "kept_count": 10,
            "input_count": 100,
            "removed_crop": 5,
            "fog_metrics": {
                "fog_inside_hull": 1,
                "fog_fraction_of_inside": 0.01,
                "transmittance_across_room": 0.95,
            },
        },
        quality={"quality_score": 80.0, "photorealism": {"mean_lpips": 0.2, "mean_psnr": 25.0}},
    )
    body = client.get("/api/quality").json()
    grades = {m["key"]: m["status"] for m in body["appearance"] + body["clarity"]}
    # LPIPS is better when lower; PSNR and transmittance when higher.
    assert grades["lpips"] == "pass"
    assert grades["psnr"] == "pass"
    assert grades["transmittance"] == "pass"
    assert body["removed"] == {"crop": 5}


def test_unit_scale_is_flagged_because_rl_needs_metres(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "build" / "visual").mkdir(parents=True)
    (workspace / "scene_manifest.json").write_text(
        json.dumps(
            {
                "warnings": [],
                "scale": {
                    "method": "floor_plane_alignment_unit_scale",
                    "meters_per_source_unit": 1.0,
                },
            }
        )
    )
    config = tmp_path / "scene.yaml"
    config.write_text(yaml.safe_dump({"name": "t", "workspace_dir": str(workspace)}))
    project_state.set_project(config, cwd=tmp_path)
    body = TestClient(create_app()).get("/api/quality").json()
    assert any("not metric" in w for w in body["warnings"])
