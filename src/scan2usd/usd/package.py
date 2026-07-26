"""Compose visual splats, collision meshes, rigid objects, and lighting into one USD stage."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP
from scan2usd.geometry.static_scene import source_to_usd_transform
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest
from scan2usd.usd.author import (
    convert_layer_with_isaac,
    matrix4d_text,
    relative_asset,
    usd_identifier,
    write_mesh_usda,
    write_object_asset_usda,
    write_object_physics_usda,
)


def _copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _artifact_path(manifest: SceneManifest, artifact_id: str) -> Path:
    artifact = manifest.artifact(artifact_id)
    if artifact is None:
        raise RuntimeError(f"Missing required artifact: {artifact_id}")
    path = Path(artifact.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _mesh_layer(
    cfg: SceneConfig,
    source: Path,
    target_dir: Path,
    name: str,
) -> Path:
    ascii_path = target_dir / f"{name}.usda"
    write_mesh_usda(source, ascii_path, meters_per_unit=cfg.usd.meters_per_unit)
    if not cfg.usd.binary_mesh_layers:
        return ascii_path
    binary_path = target_dir / f"{name}.usdc"
    converted = convert_layer_with_isaac(cfg, ascii_path, binary_path, required=True)
    if converted == binary_path:
        ascii_path.unlink(missing_ok=True)
    return converted


def _copy_material_bundle(obj: ObjectRecord, target_dir: Path) -> Path:
    if not obj.render_mesh:
        raise RuntimeError(f"{obj.instance_id} has no render mesh")
    source_dir = Path(obj.render_mesh).parent
    materials = source_dir / "materials.usda"
    if not materials.is_file():
        raise RuntimeError(f"{obj.instance_id} has no materials.usda")
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy(materials, target_dir / "materials.usda")
    textures_source = source_dir / "textures"
    textures_target = target_dir / "textures"
    if textures_target.exists():
        shutil.rmtree(textures_target)
    if textures_source.is_dir():
        shutil.copytree(textures_source, textures_target)
    return target_dir / "materials.usda"


def _build_object_package(
    cfg: SceneConfig,
    manifest: SceneManifest,
    obj: ObjectRecord,
    objects_root: Path,
) -> Path:
    if not obj.render_mesh or not obj.collision_mesh:
        raise RuntimeError(f"{obj.instance_id} is missing visual/collision geometry")
    target = objects_root / obj.instance_id
    target.mkdir(parents=True, exist_ok=True)
    geometry = _mesh_layer(cfg, Path(obj.render_mesh), target, "geometry")
    collision = _mesh_layer(cfg, Path(obj.collision_mesh), target, "collision")
    materials = _copy_material_bundle(obj, target)
    physics = write_object_physics_usda(
        target / "physics.usda",
        physics=obj.physics,
        collision_approximation=str(
            obj.physics.get("collider", cfg.physics.dynamic_collider)
        ),
    )
    asset = write_object_asset_usda(
        target / "asset.usd",
        instance_id=obj.instance_id,
        class_name=obj.class_name,
        geometry_layer=geometry,
        collision_layer=collision,
        materials_layer=materials,
        physics_layer=physics,
        default_look=cfg.usd.default_look,
    )
    manifest.register_artifact(
        artifact_id=f"object_{obj.instance_id}_usd",
        kind="usd_rigid_object",
        path=asset,
        producer="scan2usd.usd",
        metadata={"look_variants": ["baked", "pbr"]},
    )
    return asset


def _copy_environment(
    cfg: SceneConfig,
    manifest: SceneManifest,
    environment_root: Path,
) -> dict[str, Path]:
    environment_root.mkdir(parents=True, exist_ok=True)
    splat_source = _artifact_path(manifest, "environment_splat")
    splat = _copy(splat_source, environment_root / f"splat{splat_source.suffix}")
    collision = _mesh_layer(
        cfg,
        _artifact_path(manifest, "static_collision_mesh"),
        environment_root,
        "collision",
    )
    proxy = _mesh_layer(
        cfg,
        _artifact_path(manifest, "static_render_proxy"),
        environment_root,
        "proxy",
    )
    return {"splat": splat, "collision": collision, "proxy": proxy}


def _copy_lighting(manifest: SceneManifest, lighting_root: Path) -> Path:
    source = _artifact_path(manifest, "scene_lighting")
    lighting_root.mkdir(parents=True, exist_ok=True)
    target = _copy(source, lighting_root / "lighting.usda")
    # Metadata stores an absolute path in LightingEstimate; copy by basename.
    metadata_dome = manifest.artifact("scene_lighting").metadata.get("dome_texture")
    if metadata_dome:
        dome = Path(str(metadata_dome))
        if dome.is_file():
            _copy(dome, lighting_root / dome.name)
    return target


def _write_semantics(path: Path, objects: list[ObjectRecord]) -> Path:
    lines = ["#usda 1.0", "", 'over "World"', "{", '    over "Objects"', "    {"]
    for obj in objects:
        prim = usd_identifier(obj.instance_id)
        lines.extend(
            [
                f'        over "{prim}"',
                "        {",
                f'            custom string semantic:class = "{obj.class_name}"',
                f'            custom string semantic:instanceId = "{obj.instance_id}"',
                "        }",
            ]
        )
    lines.extend(["    }", "}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_scene(
    cfg: SceneConfig,
    manifest: SceneManifest,
    root: Path,
    environment: dict[str, Path],
    object_assets: dict[str, Path],
    lighting: Path,
    semantics: Path,
) -> Path:
    splat = relative_asset(environment["splat"], root)
    collision = relative_asset(environment["collision"], root)
    proxy = relative_asset(environment["proxy"], root)
    lighting_rel = relative_asset(lighting, root)
    semantics_rel = relative_asset(semantics, root)
    object_blocks: list[str] = []
    # Splat stays in COLMAP on disk; static collision/proxy meshes are already baked
    # to USD in build_static_scene. Apply T only to Splat so Isaac does not get T².
    splat_transform = source_to_usd_transform(manifest, FRAME_COLMAP)
    baked_mesh_transform = np.eye(4, dtype=np.float64)
    for obj in manifest.objects:
        if obj.instance_id not in object_assets:
            continue
        asset = relative_asset(object_assets[obj.instance_id], root)
        object_blocks.append(
            f'''        def Xform "{usd_identifier(obj.instance_id)}" (
            prepend references = @{asset}@</Object>
            variants = {{
                string look = "{cfg.usd.default_look}"
            }}
        )
        {{
            matrix4d xformOp:transform = {matrix4d_text(obj.local_to_world)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}'''
        )
    objects_text = "\n\n".join(object_blocks)
    root.write_text(
        f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = {cfg.usd.meters_per_unit:.9g}
    upAxis = "{cfg.usd.up_axis}"
    subLayers = [
        @{lighting_rel}@,
        @{semantics_rel}@
    ]
)

def Xform "World" (
    prepend variantSets = ["renderMode", "collisionDebug"]
    variants = {{
        string renderMode = "{cfg.usd.render_mode}"
        string collisionDebug = "off"
    }}
)
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}

    def Scope "Environment"
    {{
        def Xform "Splat" (
            prepend payload = @{splat}@
        )
        {{
            matrix4d xformOp:transform = {matrix4d_text(splat_transform)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }}

        def Xform "StaticCollision" (
            prepend references = @{collision}@</Geometry>
        )
        {{
            matrix4d xformOp:transform = {matrix4d_text(baked_mesh_transform)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
            over "Mesh" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
            )
            {{
                bool physics:collisionEnabled = true
                uniform token physics:approximation = "none"
                token visibility = "invisible"
            }}
        }}

        def Xform "Proxy" (
            prepend references = @{proxy}@</Geometry>
        )
        {{
            matrix4d xformOp:transform = {matrix4d_text(baked_mesh_transform)}
            uniform token[] xformOpOrder = ["xformOp:transform"]
            custom bool scan2usd:matteShadowProxy = true
            token visibility = "invisible"
        }}
    }}

    def Scope "Objects"
    {{
{objects_text}
    }}

    variantSet "renderMode" = {{
        "hybrid" {{
            over "Environment"
            {{
                over "Splat"
                {{
                    token visibility = "inherited"
                }}
                over "Proxy"
                {{
                    token visibility = "invisible"
                }}
            }}
        }}
        "splat" {{
            over "Environment"
            {{
                over "Splat"
                {{
                    token visibility = "inherited"
                }}
                over "Proxy"
                {{
                    token visibility = "invisible"
                }}
            }}
            over "Objects"
            {{
                token visibility = "invisible"
            }}
        }}
        "mesh" {{
            over "Environment"
            {{
                over "Splat"
                {{
                    token visibility = "invisible"
                }}
                over "Proxy"
                {{
                    token visibility = "inherited"
                }}
            }}
        }}
    }}

    variantSet "collisionDebug" = {{
        "off" {{
            over "Environment"
            {{
                over "StaticCollision"
                {{
                    token visibility = "invisible"
                }}
            }}
        }}
        "on" {{
            over "Environment"
            {{
                over "StaticCollision"
                {{
                    token visibility = "inherited"
                }}
            }}
        }}
    }}
}}
''',
        encoding="utf-8",
    )
    return root


def build_usd_package(
    cfg: SceneConfig,
    manifest: SceneManifest,
    *,
    output_dir: Path | None = None,
) -> Path:
    if manifest.build_mode == "production":
        manifest.require_production_ready()
    output_dir = output_dir or Path(cfg.usd.output_dir or cfg.workspace_dir / "usd")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _copy_environment(cfg, manifest, output_dir / "environment")
    lighting = _copy_lighting(manifest, output_dir / "lighting")
    object_assets = {
        obj.instance_id: _build_object_package(
            cfg,
            manifest,
            obj,
            output_dir / "objects",
        )
        for obj in manifest.objects
        if obj.movable and obj.review_state == "approved"
    }
    semantics = _write_semantics(output_dir / "semantics.usda", manifest.objects)
    root = _write_scene(
        cfg,
        manifest,
        output_dir / cfg.usd.root_filename,
        environment,
        object_assets,
        lighting,
        semantics,
    )
    manifest.register_artifact(
        artifact_id="root_usd",
        kind="isaac_scene_usd",
        path=root,
        producer="scan2usd.usd",
        metadata={
            "meters_per_unit": cfg.usd.meters_per_unit,
            "up_axis": cfg.usd.up_axis,
            "objects": len(object_assets),
        },
    )
    manifest.save(output_dir / "scene_manifest.json")
    report = {
        "root": str(root.resolve()),
        "environment": {key: str(path.resolve()) for key, path in environment.items()},
        "objects": {
            key: str(path.resolve()) for key, path in object_assets.items()
        },
        "lighting": str(lighting.resolve()),
    }
    (output_dir / "package_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return root
