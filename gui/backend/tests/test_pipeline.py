"""Pipeline state endpoint tests."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.state import project_state


def test_pipeline_state_with_dict_stages(tmp_path: Path):
    cfg = tmp_path / "scene.yaml"
    ws = tmp_path / "workspace"
    build = ws / "build"
    ns = tmp_path / "ns_data"
    build.mkdir(parents=True)
    (ns / "colmap" / "sparse" / "0").mkdir(parents=True)
    (ns / "colmap" / "sparse" / "0" / "cameras.bin").write_bytes(b"x")
    (ws / "colmap_to_usd_floor.json").write_text('{"colmap_to_usd": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}\n')
    (ws / "colmap_to_usd_metric.json").write_text('{"colmap_to_usd": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}\n')
    (ws / "masks" / "obj_001").mkdir(parents=True)
    (ws / "masks" / "obj_001" / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    cfg.write_text(
        "\n".join(
            [
                "name: test",
                f"workspace_dir: {ws}",
                f"frames_dir: {ws / 'frames'}",
                f"nerfstudio_data_dir: {ns}",
                "capture:",
                "  modality: rgb",
            ]
        )
    )
    (ws / "scene_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scene_name": "test",
                "source_config": str(cfg),
                "build_mode": "preview",
                "review_state": "approved",
                "approvals": {
                    "segmentation": {"state": "approved"},
                    "metric_transform": {"state": "approved"},
                },
                "scale": {
                    "method": "floor_alignment_plus_measured_length",
                    "approved": True,
                    "meters_per_source_unit": 0.5,
                },
                "objects": [
                    {
                        "instance_id": "obj_001",
                        "display_name": "obj_001",
                        "class_name": "obj",
                        "movable": True,
                        "review_state": "approved",
                    }
                ],
                "artifacts": [],
                "transforms": [],
                "captures": [],
                "warnings": [],
                "tool_versions": {},
            }
        )
    )
    (build / "pipeline_state.json").write_text(
        json.dumps(
            {
                "stages": {
                    "floor_alignment": {"status": "done", "at": "now"},
                    "object_proposals": {"status": "running"},
                    "mask_propagation": {"status": "completed"},
                    "legacy_flag": True,
                }
            }
        )
    )
    project_state.set_project(cfg, cwd=tmp_path)
    client = TestClient(create_app())
    r = client.get("/api/pipeline/state")
    assert r.status_code == 200, r.text
    body = r.json()
    done = set(body["stages_done"])
    assert "floor_alignment" in done
    assert "align-floor" in done
    assert "legacy_flag" in done
    assert "object_proposals" not in done
    assert "segment-usd" in done
    assert "reconstruct" in done
    assert "init-usd" in done
    assert "review" in done
    assert "apply-metric-scale" in done
    assert body["pipeline_stages"]
