from __future__ import annotations

import numpy as np
import pytest
import trimesh

from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD, compose_similarity
from scan2usd.geometry.mesh_ops import process_mesh_file
from scan2usd.geometry.static_scene import nvblox_args, source_to_usd_transform
from scan2usd.pipeline.manifest import SceneManifest, TransformRecord


def test_mesh_processing_applies_metric_transform(tmp_path):
    source = tmp_path / "cube.ply"
    trimesh.creation.box(extents=[1, 1, 1]).export(source)
    target = tmp_path / "cube_usd.ply"
    report = process_mesh_file(
        source,
        target,
        source_to_usd=compose_similarity(
            translation=np.array([1.0, 2.0, 3.0]),
            scale=2.0,
        ),
    )
    assert target.is_file()
    assert np.allclose(report.extents_m, [2.0, 2.0, 2.0])
    assert report.watertight


def test_manifest_transform_is_used_for_colmap_geometry(tmp_path):
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
    )
    transform = compose_similarity(scale=0.25)
    manifest.transforms.append(
        TransformRecord(
            source_frame=FRAME_COLMAP,
            target_frame=FRAME_USD,
            matrix=transform.tolist(),
            evidence="known length",
        )
    )
    assert np.allclose(source_to_usd_transform(manifest, FRAME_COLMAP), transform)


def test_production_missing_transform_is_rejected(tmp_path):
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
    )
    with pytest.raises(RuntimeError, match="canonical transform"):
        source_to_usd_transform(manifest, FRAME_COLMAP)


def test_nvblox_contract_includes_masks_manifest(tmp_path):
    cfg = SceneConfig()
    cfg.capture.modality = "rgbd"
    cfg.capture.depth_dir = tmp_path / "depth"
    cfg.capture.calibration_path = tmp_path / "calibration.json"
    args = nvblox_args(
        cfg,
        manifest_path=tmp_path / "manifest.json",
        output_mesh=tmp_path / "mesh.ply",
    )
    assert "--input-manifest" in args
    assert "--static-only" in args
    assert "--depth-dir" in args
    assert str(cfg.geometry.voxel_size_m) in args
