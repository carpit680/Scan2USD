from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from scan2usd.cli import app
from scan2usd.pipeline.manifest import SceneManifest


def test_init_and_approve_metric_transform(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "scene.yaml"
    config.write_text(
        "\n".join(
            [
                "name: room",
                "workspace_dir: workspace",
                "frames_dir: workspace/frames",
                "capture:",
                "  modality: rgb",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "workspace" / "frames").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(app, ["init-usd", str(config), "--mode", "production"])
    assert result.exit_code == 0, result.output
    manifest_path = tmp_path / "workspace" / "scene_manifest.json"
    assert manifest_path.is_file()

    transform_path = tmp_path / "registration.json"
    matrix = np.eye(4)
    matrix[:3, :3] *= 0.25
    transform_path.write_text(
        json.dumps(
            {
                "colmap_to_usd": matrix.tolist(),
                "registration_confidence": 0.95,
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "set-metric-transform",
            str(config),
            str(transform_path),
            "--reviewer",
            "tester",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = SceneManifest.load(manifest_path)
    assert manifest.scale.approved
    assert manifest.scale.meters_per_source_unit == 0.25
    assert manifest.approvals["metric_transform"]["state"] == "approved"


def test_apply_metric_scale_composes_floor_transform(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "scene.yaml"
    config.write_text(
        "\n".join(
            [
                "name: room",
                "workspace_dir: workspace",
                "frames_dir: workspace/frames",
                "capture:",
                "  modality: rgb",
            ]
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    (workspace / "frames").mkdir(parents=True)
    runner = CliRunner()
    assert runner.invoke(app, ["init-usd", str(config), "--mode", "preview"]).exit_code == 0

    floor = np.eye(4)
    floor[:3, 3] = [1.0, 0.0, 2.0]
    floor_path = workspace / "colmap_to_usd_floor.json"
    floor_path.write_text(
        json.dumps({"colmap_to_usd": floor.tolist(), "meters_per_source_unit": 1.0}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "apply-metric-scale",
            str(config),
            "--known-length-m",
            "0.5",
            "--source-length",
            "2.0",
            "--reviewer",
            "tester",
        ],
    )
    assert result.exit_code == 0, result.output
    metric_path = workspace / "colmap_to_usd_metric.json"
    assert metric_path.is_file()
    payload = json.loads(metric_path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["colmap_to_usd"], dtype=np.float64)
    np.testing.assert_allclose(matrix[:3, 3], [0.25, 0.0, 0.5])
    assert payload["meters_per_source_unit"] == pytest.approx(0.25)

    manifest = SceneManifest.load(workspace / "scene_manifest.json")
    assert manifest.scale.approved
    assert manifest.scale.meters_per_source_unit == pytest.approx(0.25)
    assert manifest.scale.method == "floor_alignment_plus_measured_length"
    assert manifest.approvals["metric_transform"]["state"] == "approved"

