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
