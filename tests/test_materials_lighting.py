from __future__ import annotations

import numpy as np
from PIL import Image

from scan2usd.assets.materials import build_object_materials
from scan2usd.config import SceneConfig
from scan2usd.lighting.estimate import approve_lighting, estimate_scene_lighting
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest


def test_dual_material_variants_are_authored(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    object_dir = cfg.workspace_dir / "build" / "objects" / "box_001"
    object_dir.mkdir(parents=True)
    mesh = object_dir / "geometry.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    texture = tmp_path / "atlas.png"
    gradient = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    Image.fromarray(np.dstack([gradient, gradient, gradient])).save(texture)
    obj = ObjectRecord(
        instance_id="box_001",
        display_name="Box",
        render_mesh=str(mesh),
        review_state="approved",
        physics={"template": "cardboard"},
    )
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    manifest.objects.append(obj)
    result = build_object_materials(cfg, manifest, obj, baked_texture=texture)
    text = result["materials"].read_text()
    assert 'def Material "Baked"' in text
    assert 'def Material "PBR"' in text
    assert result["normal"].is_file()
    assert obj.pbr_textures["roughness"].endswith("pbr_roughness.png")


def test_ldr_lighting_estimate_requires_review(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    images = tmp_path / "ns_data" / "images"
    images.mkdir(parents=True)
    cfg.nerfstudio_data_dir = tmp_path / "ns_data"
    for index, color in enumerate(((80, 100, 120), (120, 100, 80))):
        Image.new("RGB", (32, 24), color=color).save(images / f"{index}.jpg")
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    estimate, usda = estimate_scene_lighting(cfg, manifest)
    assert estimate.source == "ldr_capture_average"
    assert usda.is_file()
    approve_lighting(manifest, reviewer="tester")
    assert manifest.approvals["lighting"]["state"] == "approved"
