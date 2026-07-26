from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from scan2usd.assets.object_builder import finalize_object_mesh
from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_USD
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest


def _approved_scene(tmp_path, *, mode="preview"):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode=mode,
    )
    obj = ObjectRecord(
        instance_id="box_001",
        display_name="Box",
        class_name="box",
        review_state="approved",
        mask_dir=str(tmp_path / "masks"),
        observed_background_coverage=0.95,
        physics={"template": "cardboard"},
    )
    manifest.objects.append(obj)
    manifest.approve("segmentation", reviewer="tester")
    return cfg, manifest, obj


def test_finalize_object_centers_mesh_and_estimates_physics(tmp_path):
    cfg, manifest, obj = _approved_scene(tmp_path)
    raw = tmp_path / "raw.ply"
    mesh = trimesh.creation.box(extents=[0.4, 0.3, 0.2])
    mesh.apply_translation([2.0, 3.0, 1.0])
    mesh.export(raw)
    result = finalize_object_mesh(
        cfg,
        manifest,
        obj,
        raw,
        source_frame=FRAME_USD,
    )
    local = trimesh.load_mesh(result["visual"])
    assert np.allclose(local.center_mass, [0, 0, 0], atol=1e-6)
    assert obj.physics["mass_kg"] > 0
    assert obj.physics["collider"] == "convexDecomposition"
    assert np.allclose(np.asarray(obj.local_to_world)[:3, 3], [2.0, 3.0, 1.0])


def test_production_object_requires_observed_background(tmp_path):
    cfg, manifest, obj = _approved_scene(tmp_path, mode="production")
    obj.observed_background_coverage = 0.1
    raw = tmp_path / "raw.ply"
    trimesh.creation.box().export(raw)
    with pytest.raises(RuntimeError, match="clean plate"):
        finalize_object_mesh(
            cfg,
            manifest,
            obj,
            raw,
            source_frame=FRAME_USD,
        )


def test_production_object_allows_holes_when_configured(tmp_path):
    cfg, manifest, obj = _approved_scene(tmp_path, mode="production")
    cfg.qa.allow_background_holes = True
    obj.observed_background_coverage = 0.0
    raw = tmp_path / "raw.ply"
    trimesh.creation.box().export(raw)
    result = finalize_object_mesh(
        cfg,
        manifest,
        obj,
        raw,
        source_frame=FRAME_USD,
    )
    assert Path(result["visual"]).is_file()
