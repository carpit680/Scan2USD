from __future__ import annotations

import numpy as np
import pytest
import yaml

from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD, TransformGraph, compose_similarity
from scan2usd.pipeline.manifest import (
    ObjectRecord,
    ScaleEvidence,
    SceneManifest,
    TransformRecord,
)


def test_nested_config_paths_and_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": "workspace",
                "capture": {
                    "modality": "rgbd",
                    "depth_dir": "capture/depth",
                    "scale_anchor_m": 1.0,
                },
                "usd": {"output_dir": "build/usd", "up_axis": "z"},
            }
        ),
        encoding="utf-8",
    )
    cfg = SceneConfig.load(config_path)
    assert cfg.capture.modality == "rgbd"
    assert cfg.capture.depth_dir == tmp_path / "capture" / "depth"
    assert cfg.usd.output_dir == tmp_path / "build" / "usd"
    assert cfg.usd.up_axis == "Z"
    assert cfg.external["grut_python"] == "python"


def test_invalid_geometry_contract_is_rejected(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text("geometry:\n  voxel_size_m: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="voxel_size_m"):
        SceneConfig.load(path)


def test_manifest_round_trip_and_production_gate(tmp_path):
    manifest = SceneManifest.create(
        scene_name="test-room",
        source_config=tmp_path / "scene.yaml",
    )
    manifest.scale = ScaleEvidence(
        method="depth",
        meters_per_source_unit=1.0,
        confidence=1.0,
        approved=True,
    )
    manifest.transforms.append(
        TransformRecord(
            source_frame=FRAME_COLMAP,
            target_frame=FRAME_USD,
            matrix=np.eye(4).tolist(),
            evidence="metric depth registration",
        )
    )
    manifest.upsert_object(
        ObjectRecord(
            instance_id="chair_001",
            display_name="Chair 1",
            class_name="chair",
            review_state="approved",
            observed_background_coverage=0.95,
        )
    )
    manifest.approve("segmentation", reviewer="tester")
    manifest.require_production_ready()
    path = manifest.save(tmp_path / "scene_manifest.json")
    loaded = SceneManifest.load(path)
    assert loaded.get_object("chair_001").class_name == "chair"
    assert loaded.scale.is_metric()


def test_transform_graph_resolves_metric_similarity():
    graph = TransformGraph()
    colmap_to_usd = compose_similarity(
        rotation=np.eye(3),
        translation=np.array([1.0, 2.0, 3.0]),
        scale=2.0,
    )
    graph.add(FRAME_COLMAP, FRAME_USD, colmap_to_usd, evidence="scale anchor")
    point = graph.transform_points(np.array([[1.0, 0.0, 0.0]]), FRAME_COLMAP, FRAME_USD)
    assert np.allclose(point, [[3.0, 2.0, 3.0]])
    back = graph.transform_points(point, FRAME_USD, FRAME_COLMAP)
    assert np.allclose(back, [[1.0, 0.0, 0.0]])


def test_splat_command_is_registered_and_resumable():
    """
    `splat` must exist and must not depend on any USD stage.

    The point of the command is that COLMAP runs once (~22 min) while training
    is most of the wall-clock and is what gets iterated on, so a re-run after a
    config change has to resume rather than repeat.
    """
    import inspect

    from scan2usd import cli

    assert hasattr(cli, "splat")
    source = inspect.getsource(cli.splat)

    # Resume points for the two cheap-to-check stages. Training is the expensive
    # one and is checked behaviourally in the test below — a grep here passed
    # while the call retrained on every run.
    assert "frames_dir_has_images" in source
    assert "find_ns_colmap_sparse" in source

    # Mode is pinned, because chaining the individual commands on their default
    # arguments raises: init-usd defaults to production, align-floor to preview.
    assert 'build_mode="preview"' in source

    # None of the USD stages may appear on this path.
    for forbidden in (
        "build_static_scene",
        "build_usd_package",
        "estimate_scene_lighting",
        "ns_train_splatfacto",
        "validate",
    ):
        assert forbidden not in source, f"{forbidden} must not run on the splat path"


def _splat_harness(tmp_path, monkeypatch, *, calls):
    """
    Drive `splat` with every stage but training already satisfied.

    Frames, COLMAP and the TXT export are made to look complete so the only
    decision left is the one under test: does a second run retrain?
    """
    import types

    from scan2usd import cli
    from scan2usd.reconstruction import grut

    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "workspace"
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00001.jpg").write_bytes(b"")
    sparse = tmp_path / "ns_data" / "colmap" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(b"")
    txt = workspace / "colmap_txt"
    txt.mkdir(parents=True)
    (txt / "images.txt").write_text("", encoding="utf-8")
    visual = workspace / "build" / "visual"
    visual.mkdir(parents=True)
    (visual / "environment_splat.usd").write_bytes(b"stub")

    config_path = tmp_path / "scene.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": "workspace",
                "frames_dir": "frames",
                "nerfstudio_data_dir": "ns_data",
                "colmap_txt_dir": "workspace/colmap_txt",
            }
        ),
        encoding="utf-8",
    )

    def _train(cfg, manifest, **kwargs):
        calls.append("train")
        return visual / "environment_splat.usd"

    monkeypatch.setattr(grut, "export_environment_particlefield", _train)
    monkeypatch.setattr(cli, "find_ns_colmap_sparse", lambda _dir: sparse)
    monkeypatch.setattr(cli, "_check_registration_rate", lambda cfg, threshold: (1, 1))
    monkeypatch.setattr(cli, "ns_process_data_images", lambda *a, **k: pytest.fail("re-ran COLMAP"))

    class _Orchestrator:
        """Real resume semantics, without the manifest/USD machinery around them."""

        def __init__(self, cfg, config_path, build_mode):
            assert build_mode == "preview"
            self.manifest = types.SimpleNamespace(save=lambda path: path)
            self.manifest_path = workspace / "scene_manifest.json"
            self.state = {"stages": {}}

        def align_floor(self):
            calls.append("floor")

        def _artifact_ready(self, name):
            return (visual / "environment_splat.usd").is_file()

        def run_stage(self, name, operation, *, force=False, ready=None):
            calls.append(f"run_stage:{name}")
            stage = self.state["stages"].get(name, {})
            if not force and stage.get("status") == "completed" and (ready is None or ready()):
                return None
            result = operation()
            self.state["stages"][name] = {"status": "completed"}
            return result

    import scan2usd.pipeline.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "PipelineOrchestrator", _Orchestrator)
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kwargs: calls.append("export_ply") or None
    )
    monkeypatch.setattr(
        "scan2usd.reconstruction.external_cli.resolve_external_command",
        lambda cfg, name, default=None, required=False: ["python"],
    )
    return config_path, _Orchestrator


def test_splat_does_not_retrain_when_the_splat_is_already_there(tmp_path, monkeypatch):
    """
    Training is ~85% of the run, so the resume decision is the whole point.

    An earlier version called ``export_environment_particlefield`` directly and
    only cleared manifest state under ``--force-train``, which meant every
    invocation retrained — the exact opposite of the intent, and invisible to a
    test that greps the source for existence checks.
    """
    from typer.testing import CliRunner

    from scan2usd import cli

    calls: list[str] = []
    config_path, orchestrator_class = _splat_harness(tmp_path, monkeypatch, calls=calls)
    runner = CliRunner()

    first = runner.invoke(cli.app, ["splat", str(config_path)])
    assert first.exit_code == 0, first.output
    assert calls.count("train") == 1

    # Second run: the stage is completed and the artifact exists, so training
    # must be skipped while the preview is still produced.
    completed = {"stages": {"visual_particlefield": {"status": "completed"}}}
    original_init = orchestrator_class.__init__

    def _resumed_init(self, cfg, path, build_mode):
        original_init(self, cfg, path, build_mode)
        self.state = completed

    monkeypatch.setattr(orchestrator_class, "__init__", _resumed_init)
    calls.clear()
    second = runner.invoke(cli.app, ["splat", str(config_path)])
    assert second.exit_code == 0, second.output
    assert "train" not in calls, "re-ran training with an existing splat"
    assert "export_ply" in calls, "resume must still produce the preview"

    # …unless asked to.
    calls.clear()
    forced = runner.invoke(cli.app, ["splat", str(config_path), "--force-train"])
    assert forced.exit_code == 0, forced.output
    assert calls.count("train") == 1


def test_splat_reaches_the_preview_the_gui_serves():
    """The command must write exactly the file the Preview tab reads."""
    import inspect

    from scan2usd import cli

    source = inspect.getsource(cli.splat)
    # gui/backend/scan2usd_gui/routes/quality.py serves <ws>/build/visual/preview.ply
    assert '"preview.ply"' in source
    assert "export_splat_ply.py" in source
