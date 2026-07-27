"""Pipeline stage skip / ready checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from scan2usd.pipeline.orchestrator import PipelineOrchestrator


def _orch(tmp_path: Path, *, masks_dir: Path | None = None) -> PipelineOrchestrator:
    cfg = MagicMock()
    cfg.workspace_dir = tmp_path
    cfg.segmentation.masks_dir = masks_dir or (tmp_path / "masks")
    cfg.nerfstudio_data_dir = tmp_path / "ns_data"
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.cfg = cfg
    orch.manifest = MagicMock()
    orch.build_root = tmp_path / "build"
    orch.state_path = tmp_path / "build" / "pipeline_state.json"
    orch.manifest_path = tmp_path / "scene_manifest.json"
    orch.state = {"schema_version": "1.0", "stages": {}}
    return orch


def test_instance_masks_ready_false_without_pngs(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    report = tmp_path / "build" / "segmentation" / "mask_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    art = MagicMock()
    art.path = str(report)
    orch.manifest.artifact.return_value = art
    (tmp_path / "masks").mkdir()
    assert orch._instance_masks_ready() is False


def test_instance_masks_ready_true_with_png(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    report = tmp_path / "build" / "segmentation" / "mask_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    art = MagicMock()
    art.path = str(report)
    orch.manifest.artifact.return_value = art
    mask = tmp_path / "masks" / "laptop_001" / "frame_00001.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"x")
    assert orch._instance_masks_ready() is True


def test_baked_ready_false_on_transform_hash_mismatch(tmp_path: Path) -> None:
    """Meshes baked under an old COLMAP→USD transform must be re-baked."""
    from scan2usd.pipeline.manifest import SceneManifest, TransformRecord

    orch = _orch(tmp_path)
    mesh = tmp_path / "build" / "geometry" / "static_collision.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"ply")
    manifest = SceneManifest(scene_name="t", source_config="cfg.yaml", build_mode="preview")
    manifest.transforms.append(
        TransformRecord(
            source_frame="colmap_world",
            target_frame="usd_world_z_up_meters",
            matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
    )
    orch.manifest = manifest
    manifest.register_artifact(
        artifact_id="static_collision_mesh",
        kind="triangle_collision_mesh",
        path=mesh,
        producer="test",
        metadata={"transform_hash": manifest.colmap_to_usd_hash()},
    )
    assert orch._baked_ready("static_collision_mesh") is True

    # Re-running floor alignment changes the transform → artifact goes stale.
    manifest.transforms[0].matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0.5],
        [0, 0, 0, 1],
    ]
    assert orch._baked_ready("static_collision_mesh") is False


def test_baked_ready_false_for_legacy_unstamped_artifact(tmp_path: Path) -> None:
    """Artifacts baked before hash stamping rebuild once so they get stamped."""
    from scan2usd.pipeline.manifest import SceneManifest, TransformRecord

    orch = _orch(tmp_path)
    mesh = tmp_path / "static_proxy.ply"
    mesh.write_bytes(b"ply")
    manifest = SceneManifest(scene_name="t", source_config="cfg.yaml", build_mode="preview")
    manifest.transforms.append(
        TransformRecord(
            source_frame="colmap_world",
            target_frame="usd_world_z_up_meters",
            matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
    )
    orch.manifest = manifest
    manifest.register_artifact(
        artifact_id="static_render_proxy",
        kind="render_proxy_mesh",
        path=mesh,
        producer="test",
        metadata={},
    )
    assert orch._baked_ready("static_render_proxy") is False


def test_baked_ready_true_without_any_transform(tmp_path: Path) -> None:
    """No COLMAP→USD transform yet (early preview) — do not force rebuild loops."""
    from scan2usd.pipeline.manifest import SceneManifest

    orch = _orch(tmp_path)
    mesh = tmp_path / "static_proxy.ply"
    mesh.write_bytes(b"ply")
    manifest = SceneManifest(scene_name="t", source_config="cfg.yaml", build_mode="preview")
    orch.manifest = manifest
    manifest.register_artifact(
        artifact_id="static_render_proxy",
        kind="render_proxy_mesh",
        path=mesh,
        producer="test",
        metadata={},
    )
    assert orch._baked_ready("static_render_proxy") is True


def test_run_stage_reruns_when_ready_false(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    orch.state_path.parent.mkdir(parents=True)
    orch.manifest_path.write_text("{}")
    orch.manifest.save = MagicMock()
    orch.state["stages"]["mask_propagation"] = {"status": "completed"}
    calls = {"n": 0}

    def op():
        calls["n"] += 1
        return "ok"

    result = orch.run_stage("mask_propagation", op, force=False, ready=lambda: False)
    assert result == "ok"
    assert calls["n"] == 1
