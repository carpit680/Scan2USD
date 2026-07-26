"""Per-instance rigid-object reconstruction, cleanup, and collision preparation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scan2usd.assets.physics import PhysicalProperties, estimate_physical_properties
from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP
from scan2usd.geometry.mesh_ops import (
    clean_mesh,
    load_mesh,
    mesh_report,
    simplify_mesh,
    transform_mesh,
)
from scan2usd.geometry.static_scene import source_to_usd_transform
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest
from scan2usd.reconstruction.external_cli import ExternalToolAdapter, resolve_external_command


def reconstruction_args(
    cfg: SceneConfig,
    obj: ObjectRecord,
    *,
    output_mesh: Path,
) -> list[str]:
    if not obj.mask_dir:
        raise RuntimeError(f"{obj.instance_id} has no reviewed mask directory")
    args = [
        "--images",
        str((cfg.nerfstudio_data_dir / "images").resolve()),
        "--masks",
        str(Path(obj.mask_dir).resolve()),
        "--colmap",
        str((cfg.nerfstudio_data_dir / "colmap" / "sparse" / "0").resolve()),
        "--instance-id",
        obj.instance_id,
        "--output-mesh",
        str(output_mesh.resolve()),
        "--texture-resolution",
        str(cfg.materials.texture_resolution),
    ]
    detail_dir = cfg.capture.object_capture_dirs.get(obj.instance_id)
    if detail_dir is not None:
        args.extend(["--detail-capture", str(detail_dir.resolve())])
    return args


def _require_object_gate(cfg: SceneConfig, manifest: SceneManifest, obj: ObjectRecord) -> None:
    if obj.review_state != "approved":
        raise RuntimeError(f"Object {obj.instance_id} has not been approved")
    if (
        manifest.build_mode == "production"
        and obj.movable
        and not cfg.qa.allow_background_holes
        and obj.observed_background_coverage < cfg.qa.min_background_coverage
        and cfg.capture.clean_plate_dir is None
    ):
        raise RuntimeError(
            f"{obj.instance_id} would leave a splat ghost/hole; acquire a clean plate "
            "(or set qa.allow_background_holes for development)"
        )


def _localize_at_center_of_mass(mesh, properties: PhysicalProperties):
    center = np.asarray(properties.center_of_mass_m, dtype=np.float64)
    local = mesh.copy()
    local.apply_translation(-center)
    local_to_world = np.eye(4, dtype=np.float64)
    local_to_world[:3, 3] = center
    physics = properties.to_dict()
    physics["center_of_mass_m"] = [0.0, 0.0, 0.0]
    return local, local_to_world, physics


def finalize_object_mesh(
    cfg: SceneConfig,
    manifest: SceneManifest,
    obj: ObjectRecord,
    raw_mesh: Path,
    *,
    source_frame: str = FRAME_COLMAP,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Canonicalize and localize one render mesh; create an Isaac decomposition source."""
    _require_object_gate(cfg, manifest, obj)
    output_dir = output_dir or (cfg.workspace_dir / "build" / "objects" / obj.instance_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = clean_mesh(load_mesh(raw_mesh), fill_small_holes=True)
    mesh = transform_mesh(mesh, source_to_usd_transform(manifest, source_frame))
    mesh = clean_mesh(mesh, fill_small_holes=True)
    if manifest.build_mode == "production" and not mesh.is_watertight:
        raise RuntimeError(
            f"{obj.instance_id} render mesh is not watertight; correct masks/detail capture "
            "instead of silently substituting a convex hull"
        )

    template = str(obj.physics.get("template", "generic"))
    properties = estimate_physical_properties(mesh, cfg.physics, template=template)
    local_mesh, local_to_world, local_physics = _localize_at_center_of_mass(mesh, properties)
    visual_path = output_dir / "geometry.obj"
    local_mesh.export(visual_path)

    collision_mesh = simplify_mesh(
        local_mesh,
        min(20_000, max(256, cfg.geometry.target_object_triangles // 4)),
    )
    collision_path = output_dir / "collision_source.ply"
    collision_mesh.export(collision_path)

    obj.local_to_world = local_to_world.tolist()
    obj.render_mesh = str(visual_path.resolve())
    obj.collision_mesh = str(collision_path.resolve())
    obj.physics.update(local_physics)
    obj.physics["approved"] = False
    manifest.upsert_object(obj)
    manifest.register_artifact(
        artifact_id=f"object_{obj.instance_id}_visual",
        kind="rigid_object_render_mesh",
        path=visual_path,
        producer="scan2usd.object_builder",
        metadata={"report": asdict(mesh_report(local_mesh, visual_path))},
    )
    manifest.register_artifact(
        artifact_id=f"object_{obj.instance_id}_collision",
        kind="convex_decomposition_source",
        path=collision_path,
        producer="scan2usd.object_builder",
        metadata={
            "approximation": cfg.physics.dynamic_collider,
            "vhacd_max_hulls": cfg.physics.vhacd_max_hulls,
            "vhacd_resolution": cfg.physics.vhacd_resolution,
            "report": asdict(mesh_report(collision_mesh, collision_path)),
        },
    )
    report = {
        "instance_id": obj.instance_id,
        "source_mesh": str(raw_mesh.resolve()),
        "source_frame": source_frame,
        "visual_mesh": str(visual_path.resolve()),
        "collision_source": str(collision_path.resolve()),
        "local_to_world": obj.local_to_world,
        "physics": obj.physics,
    }
    (output_dir / "object_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"visual": visual_path, "collision": collision_path}


def reconstruct_object(
    cfg: SceneConfig,
    manifest: SceneManifest,
    instance_id: str,
) -> dict[str, Path]:
    """Run the configured masked object reconstructor and finalize its assets."""
    obj = manifest.get_object(instance_id)
    _require_object_gate(cfg, manifest, obj)
    output_dir = cfg.workspace_dir / "build" / "objects" / instance_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_mesh = output_dir / "reconstruction" / "textured.obj"
    raw_mesh.parent.mkdir(parents=True, exist_ok=True)
    prefix = resolve_external_command(
        cfg,
        "object_reconstruction_runner",
        default="",
        required=True,
    )
    assert prefix is not None
    adapter = ExternalToolAdapter("object_reconstruction", prefix)
    adapter.run(*reconstruction_args(cfg, obj, output_mesh=raw_mesh))
    if not raw_mesh.is_file():
        candidates = sorted(raw_mesh.parent.glob("*.obj")) + sorted(
            raw_mesh.parent.glob("*.ply")
        )
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Object reconstructor did not emit one mesh under {raw_mesh.parent}"
            )
        raw_mesh = candidates[0]
    return finalize_object_mesh(
        cfg,
        manifest,
        obj,
        raw_mesh,
        source_frame=FRAME_COLMAP,
        output_dir=output_dir,
    )
