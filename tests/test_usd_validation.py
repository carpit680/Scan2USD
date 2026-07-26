from __future__ import annotations

import pytest

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import SceneManifest
from scan2usd.usd.validate import validate_usd


def test_preview_validation_reports_skipped_external_checks(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.colmap_txt_dir = tmp_path / "missing_colmap"
    cfg.usd.output_dir = tmp_path / "usd"
    root = cfg.usd.output_dir / "scene.usd"
    root.parent.mkdir(parents=True)
    root.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    manifest.register_artifact(
        artifact_id="root_usd",
        kind="isaac_scene_usd",
        path=root,
        producer="test",
    )
    report = validate_usd(
        cfg,
        manifest,
        run_isaac=False,
    )
    assert report["usable"]
    assert (cfg.usd.output_dir / "build_report.json").is_file()


def test_production_validation_requires_heldout_and_isaac(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.colmap_txt_dir = tmp_path / "missing_colmap"
    cfg.usd.output_dir = tmp_path / "usd"
    root = cfg.usd.output_dir / "scene.usd"
    root.parent.mkdir(parents=True)
    root.write_text('#usda 1.0\n\ndef Xform "World" {}\n', encoding="utf-8")
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="production",
    )
    manifest.register_artifact(
        artifact_id="root_usd",
        kind="isaac_scene_usd",
        path=root,
        producer="test",
    )
    with pytest.raises(RuntimeError, match="USD validation failed"):
        validate_usd(cfg, manifest, run_isaac=False)
