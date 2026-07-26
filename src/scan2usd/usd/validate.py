"""Quality gates for registration, assets, photorealism, USD, and Isaac physics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scan2usd.config import SceneConfig
from scan2usd.eval.photorealism import evaluate_held_out_renders
from scan2usd.geometry.frames import FRAME_COLMAP, apply_transform
from scan2usd.geometry.mesh_ops import load_mesh
from scan2usd.geometry.static_scene import source_to_usd_transform
from scan2usd.pipeline.manifest import SceneManifest
from scan2usd.reconstruction.colmap_io import parse_points3d_txt
from scan2usd.reconstruction.external_cli import ExternalToolAdapter, resolve_external_command


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    required: bool
    details: dict[str, Any]


def _nearest_vertex_distances(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(vertices) == 0:
        return np.empty(0)
    try:
        from scipy.spatial import cKDTree

        distances, _indices = cKDTree(vertices).query(points, k=1)
        return np.asarray(distances)
    except ImportError:
        output = np.empty(len(points), dtype=np.float64)
        for start in range(0, len(points), 256):
            chunk = points[start : start + 256]
            squared = np.sum((chunk[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
            output[start : start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
        return output


def _registration_check(
    cfg: SceneConfig,
    manifest: SceneManifest,
) -> ValidationCheck:
    proxy_artifact = manifest.artifact("static_render_proxy")
    points_path = cfg.colmap_txt_dir / "points3D.txt"
    required = manifest.build_mode == "production"
    if proxy_artifact is None or not points_path.is_file():
        return ValidationCheck(
            "splat_proxy_registration",
            not required,
            required,
            {"skipped": "static proxy or COLMAP points are missing"},
        )
    _ids, points = parse_points3d_txt(points_path)
    if len(points) > 5_000:
        indices = np.linspace(0, len(points) - 1, 5_000, dtype=np.int64)
        points = points[indices]
    points_usd = apply_transform(
        points,
        source_to_usd_transform(manifest, FRAME_COLMAP),
    )
    proxy = load_mesh(Path(proxy_artifact.path))
    vertices = np.asarray(proxy.vertices)
    if len(vertices) > 200_000:
        indices = np.linspace(0, len(vertices) - 1, 200_000, dtype=np.int64)
        vertices = vertices[indices]
    distances = _nearest_vertex_distances(points_usd, vertices)
    median = float(np.median(distances)) if len(distances) else None
    p95 = float(np.percentile(distances, 95)) if len(distances) else None
    passed = median is not None and median <= cfg.qa.max_registration_error_m
    return ValidationCheck(
        "splat_proxy_registration",
        passed,
        required,
        {
            "samples": len(distances),
            "median_vertex_distance_m": median,
            "p95_vertex_distance_m": p95,
            "threshold_m": cfg.qa.max_registration_error_m,
            "note": (
                "Compares COLMAP points transformed by the manifest COLMAP→USD matrix "
                "against on-disk proxy vertices (already baked to USD). StaticCollision/"
                "Proxy parent xforms are identity so Isaac does not apply T twice."
            ),
        },
    )


def _object_checks(
    cfg: SceneConfig,
    manifest: SceneManifest,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    required = manifest.build_mode == "production"
    for obj in manifest.objects:
        if not obj.movable or obj.review_state != "approved":
            continue
        files = {
            "render_mesh": obj.render_mesh,
            "collision_mesh": obj.collision_mesh,
            "baked_texture": obj.baked_texture,
            **{f"pbr_{key}": value for key, value in obj.pbr_textures.items()},
        }
        missing = [name for name, value in files.items() if not value or not Path(value).is_file()]
        checks.append(
            ValidationCheck(
                f"object_files:{obj.instance_id}",
                not missing,
                required,
                {"missing": missing},
            )
        )
        if obj.render_mesh and Path(obj.render_mesh).is_file():
            visual = load_mesh(Path(obj.render_mesh))
            uv = getattr(visual.visual, "uv", None)
            checks.append(
                ValidationCheck(
                    f"object_visual_mesh:{obj.instance_id}",
                    bool(visual.is_watertight) and uv is not None and len(uv) == len(visual.vertices),
                    required,
                    {
                        "watertight": bool(visual.is_watertight),
                        "vertices": len(visual.vertices),
                        "faces": len(visual.faces),
                        "has_vertex_uv": uv is not None and len(uv) == len(visual.vertices),
                    },
                )
            )
        if obj.collision_mesh and Path(obj.collision_mesh).is_file():
            collision = load_mesh(Path(obj.collision_mesh))
            checks.append(
                ValidationCheck(
                    f"object_collider:{obj.instance_id}",
                    len(collision.faces) <= cfg.qa.max_dynamic_collider_faces,
                    required,
                    {
                        "faces": len(collision.faces),
                        "max_faces": cfg.qa.max_dynamic_collider_faces,
                        "approximation": obj.physics.get("collider"),
                    },
                )
            )
        physics_keys = {
            "mass_kg",
            "diagonal_inertia_kg_m2",
            "friction",
            "restitution",
        }
        checks.append(
            ValidationCheck(
                f"object_physics:{obj.instance_id}",
                not (physics_keys - set(obj.physics))
                and (not required or bool(obj.physics.get("approved"))),
                required,
                {
                    "missing": sorted(physics_keys - set(obj.physics)),
                    "approved": bool(obj.physics.get("approved")),
                },
            )
        )
        ghost_free = (
            obj.observed_background_coverage >= cfg.qa.min_background_coverage
            or cfg.capture.clean_plate_dir is not None
        )
        checks.append(
            ValidationCheck(
                f"object_background:{obj.instance_id}",
                ghost_free,
                required,
                {
                    "coverage": obj.observed_background_coverage,
                    "threshold": cfg.qa.min_background_coverage,
                    "clean_plate": str(cfg.capture.clean_plate_dir or ""),
                },
            )
        )
    return checks


def _depth_collider_check(cfg: SceneConfig, manifest: SceneManifest) -> ValidationCheck:
    required = manifest.build_mode == "production" and cfg.capture.modality in {
        "rgbd",
        "lidar",
    }
    artifact = manifest.artifact("static_collision_mesh")
    validation = artifact.metadata.get("depth_validation") if artifact else None
    if not isinstance(validation, dict):
        return ValidationCheck(
            "depth_to_collider",
            not required,
            required,
            {"skipped": "dense backend did not emit depth_validation.json"},
        )
    median = validation.get("median_depth_error_m")
    passed = median is not None and float(median) <= cfg.qa.max_depth_error_m
    return ValidationCheck(
        "depth_to_collider",
        passed,
        required,
        {
            **validation,
            "threshold_m": cfg.qa.max_depth_error_m,
        },
    )


def _run_isaac_validation(
    cfg: SceneConfig,
    root_stage: Path,
    output_path: Path,
    *,
    required: bool,
) -> ValidationCheck:
    prefix = resolve_external_command(
        cfg,
        "isaac_python",
        default="python.sh",
        required=required,
    )
    if prefix is None:
        return ValidationCheck(
            "isaac_headless",
            True,
            False,
            {"skipped": "Isaac Python is not configured"},
        )
    script = Path(__file__).resolve().parents[3] / "tools" / "isaac" / "validate_scene.py"
    adapter = ExternalToolAdapter("isaac_python", prefix)
    result = adapter.run(
        str(script),
        "--stage",
        str(root_stage.resolve()),
        "--output",
        str(output_path.resolve()),
        check=False,
        capture_output=True,
    )
    report = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.is_file()
        else {"errors": [result.stderr or result.stdout or "Isaac validator produced no report"]}
    )
    return ValidationCheck(
        "isaac_headless",
        bool(report.get("passed")) and result.returncode == 0,
        required,
        report,
    )


def validate_usd(
    cfg: SceneConfig,
    manifest: SceneManifest,
    *,
    held_out_render_dir: Path | None = None,
    run_isaac: bool = True,
    fail_on_error: bool = True,
) -> dict[str, Any]:
    root_artifact = manifest.artifact("root_usd")
    if root_artifact is None:
        raise RuntimeError("No root_usd artifact; run package-usd first")
    root = Path(root_artifact.path)
    build_report = Path(cfg.usd.output_dir or cfg.workspace_dir / "usd") / "build_report.json"
    checks: list[ValidationCheck] = [
        ValidationCheck(
            "root_stage",
            root.is_file(),
            True,
            {"path": str(root.resolve())},
        ),
        _registration_check(cfg, manifest),
        _depth_collider_check(cfg, manifest),
        *_object_checks(cfg, manifest),
    ]

    heldout_spec = cfg.workspace_dir / "build" / "grut_dataset" / "held_out.json"
    heldout_required = manifest.build_mode == "production" and cfg.qa.require_held_out_renders
    if held_out_render_dir is not None and heldout_spec.is_file():
        metrics = evaluate_held_out_renders(
            heldout_spec,
            cfg.workspace_dir / "build" / "grut_dataset" / "images",
            held_out_render_dir,
            output_path=Path(cfg.usd.output_dir or cfg.workspace_dir / "usd")
            / "photorealism_report.json",
        )
        checks.append(
            ValidationCheck(
                "held_out_photorealism",
                metrics["evaluated"] == metrics["expected"] and metrics["evaluated"] > 0,
                heldout_required,
                metrics,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                "held_out_photorealism",
                not heldout_required,
                heldout_required,
                {"skipped": "provide --held-out-renders from Isaac NuRec rendering"},
            )
        )

    isaac_required = manifest.build_mode == "production" and cfg.qa.require_isaac_validation
    if run_isaac:
        checks.append(
            _run_isaac_validation(
                cfg,
                root,
                Path(cfg.usd.output_dir or cfg.workspace_dir / "usd")
                / "isaac_validation.json",
                required=isaac_required,
            )
        )
    else:
        checks.append(
            ValidationCheck(
                "isaac_headless",
                not isaac_required,
                isaac_required,
                {"skipped": "disabled by caller"},
            )
        )

    failed_required = [check.name for check in checks if check.required and not check.passed]
    report = {
        "schema_version": "1.0",
        "scene": manifest.scene_name,
        "root_usd": str(root.resolve()),
        "usable": not failed_required,
        "failed_required_checks": failed_required,
        "checks": [asdict(check) for check in checks],
        "limitations": [
            "ParticleField radiance is capture-lit and not a fully relightable PBR surface.",
            "PBR object de-lighting and material maps are estimates until reviewed.",
            "RGB-only collision accuracy depends on the supplied metric registration.",
        ],
    }
    build_report.parent.mkdir(parents=True, exist_ok=True)
    build_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest.register_artifact(
        artifact_id="build_report",
        kind="usd_validation_report",
        path=build_report,
        producer="scan2usd.validate",
        metadata={"usable": report["usable"]},
    )
    if fail_on_error and failed_required:
        raise RuntimeError("USD validation failed: " + ", ".join(failed_required))
    return report
