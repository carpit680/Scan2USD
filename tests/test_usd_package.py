from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

from scan2usd.assets.materials import build_object_materials
from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest, TransformRecord
from scan2usd.usd.author import write_mesh_usda
from scan2usd.usd.package import build_usd_package


def test_mesh_usda_contains_triangle_topology(tmp_path):
    source = tmp_path / "cube.ply"
    trimesh.creation.box().export(source)
    layer = write_mesh_usda(source, tmp_path / "geometry.usda")
    text = layer.read_text()
    assert 'def Mesh "Mesh"' in text
    assert "faceVertexIndices" in text
    assert 'upAxis = "Z"' in text


def test_build_layered_preview_package(tmp_path):
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.usd.output_dir = tmp_path / "usd"
    cfg.usd.binary_mesh_layers = False
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    splat = artifacts / "splat.usda"
    splat.write_text('#usda 1.0\n\ndef Xform "Splat" {}\n', encoding="utf-8")
    static = artifacts / "static.ply"
    trimesh.creation.box(extents=[5, 5, 0.1]).export(static)
    manifest.register_artifact(
        artifact_id="environment_splat",
        kind="usd_particle_field",
        path=splat,
        producer="test",
    )
    manifest.register_artifact(
        artifact_id="static_collision_mesh",
        kind="triangle_collision_mesh",
        path=static,
        producer="test",
    )
    manifest.register_artifact(
        artifact_id="static_render_proxy",
        kind="render_proxy_mesh",
        path=static,
        producer="test",
    )

    lighting_dir = artifacts / "lighting"
    lighting_dir.mkdir()
    dome = lighting_dir / "dome.png"
    Image.new("RGB", (8, 4), color=(100, 100, 100)).save(dome)
    lighting = lighting_dir / "lighting.usda"
    lighting.write_text('#usda 1.0\n\ndef Scope "Lighting" {}\n', encoding="utf-8")
    manifest.register_artifact(
        artifact_id="scene_lighting",
        kind="usd_lighting",
        path=lighting,
        producer="test",
        metadata={"dome_texture": str(dome)},
    )

    object_dir = artifacts / "object"
    object_dir.mkdir()
    geometry = object_dir / "geometry.obj"
    collision = object_dir / "collision.ply"
    cube = trimesh.creation.box(extents=[0.4, 0.3, 0.2])
    cube.export(geometry)
    cube.export(collision)
    texture = object_dir / "atlas.png"
    Image.new("RGB", (16, 16), color=(150, 100, 50)).save(texture)
    obj = ObjectRecord(
        instance_id="box_001",
        display_name="Box",
        class_name="box",
        review_state="approved",
        render_mesh=str(geometry),
        collision_mesh=str(collision),
        physics={
            "template": "cardboard",
            "mass_kg": 1.0,
            "diagonal_inertia_kg_m2": [0.1, 0.1, 0.1],
            "principal_axes": np.eye(3).tolist(),
            "friction": 0.5,
            "restitution": 0.05,
            "collider": "convexDecomposition",
        },
    )
    manifest.objects.append(obj)
    build_object_materials(cfg, manifest, obj, baked_texture=texture)
    # Non-identity COLMAP→USD: applied to Splat only; baked meshes stay identity.
    colmap_to_usd = [
        [1.0, 0.0, 0.0, 1.25],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    manifest.transforms.append(
        TransformRecord(
            source_frame=FRAME_COLMAP,
            target_frame=FRAME_USD,
            matrix=colmap_to_usd,
            confidence=0.9,
            evidence="test",
        )
    )
    root = build_usd_package(cfg, manifest)
    text = root.read_text()
    assert root.is_file()
    assert 'def PhysicsScene "PhysicsScene"' in text
    assert 'def Xform "Splat"' in text
    assert 'def Xform "box_001"' in text
    assert (cfg.usd.output_dir / "objects" / "box_001" / "asset.usd").is_file()
    # Splat parent carries COLMAP→USD as a USD row-vector matrix (translation in last row).
    assert "1.25" in text
    splat_idx = text.find('def Xform "Splat"')
    collision_idx = text.find('def Xform "StaticCollision"')
    proxy_idx = text.find('def Xform "Proxy"')
    splat_block = text[splat_idx:collision_idx]
    collision_block = text[collision_idx:proxy_idx]
    proxy_block = text[proxy_idx : text.find('def Scope "Objects"')]
    assert "(1.25, 0, 0.5, 1)" in splat_block.replace(".0", "") or "(1.25, 0, 0.5, 1)" in splat_block
    assert "1.25" in splat_block
    assert "(1, 0, 0, 0)" in collision_block.replace(".0", "")
    assert "(1, 0, 0, 0)" in proxy_block.replace(".0", "")
    assert "1.25" not in collision_block
    assert "1.25" not in proxy_block


def test_matrix4d_text_uses_usd_row_vector_convention() -> None:
    from scan2usd.usd.author import matrix4d_text

    # Column-vector similarity: translate by (1, 2, 3).
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    text = matrix4d_text(matrix)
    assert "(1, 0, 0, 0)" in text.replace(".0", "")
    assert "(1, 2, 3, 1)" in text.replace(".0", "") or "(1, 2, 3, 1)" in text
