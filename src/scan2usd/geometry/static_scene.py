"""Dense static-scene reconstruction and collision proxy generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD, TransformGraph
from scan2usd.geometry.mesh_ops import (
    clean_mesh,
    load_mesh,
    mesh_report,
    process_mesh_file,
    simplify_mesh,
    split_static_mesh,
)
from scan2usd.pipeline.manifest import SceneManifest
from scan2usd.reconstruction.external_cli import (
    ExternalToolAdapter,
    external_tool,
    resolve_colmap,
)


def _manifest_graph(manifest: SceneManifest) -> TransformGraph:
    graph = TransformGraph()
    for transform in manifest.transforms:
        graph.add(
            transform.source_frame,
            transform.target_frame,
            transform.matrix,
            evidence=transform.evidence,
            confidence=transform.confidence,
        )
    return graph


def source_to_usd_transform(
    manifest: SceneManifest,
    source_frame: str,
) -> np.ndarray:
    if source_frame == FRAME_USD:
        return np.eye(4)
    try:
        return _manifest_graph(manifest).resolve(source_frame, FRAME_USD)
    except KeyError:
        if manifest.build_mode == "preview":
            manifest.warnings.append(
                f"No {source_frame}→{FRAME_USD} transform; preview uses identity/non-metric scale"
            )
            return np.eye(4)
        raise RuntimeError(
            f"Production geometry lacks canonical transform {source_frame}→{FRAME_USD}"
        ) from None


def require_metric_geometry(cfg: SceneConfig, manifest: SceneManifest) -> None:
    if manifest.build_mode != "production":
        return
    if not manifest.scale.is_metric():
        raise RuntimeError("Production collision geometry requires approved metric scale")
    if (
        cfg.capture.modality == "rgb"
        and cfg.geometry.rgb_only_requires_scale
        and manifest.scale.method == "unknown"
    ):
        raise RuntimeError("RGB-only geometry requires a scale anchor")


def nvblox_args(
    cfg: SceneConfig,
    *,
    manifest_path: Path,
    output_mesh: Path,
) -> list[str]:
    args = [
        "--input-manifest",
        str(manifest_path.resolve()),
        "--output-mesh",
        str(output_mesh.resolve()),
        "--validation-report",
        str((output_mesh.parent / "depth_validation.json").resolve()),
        "--voxel-size",
        str(cfg.geometry.voxel_size_m),
        "--static-only",
    ]
    if cfg.capture.depth_dir is not None:
        args.extend(["--depth-dir", str(cfg.capture.depth_dir.resolve())])
    if cfg.capture.lidar_path is not None:
        args.extend(["--lidar", str(cfg.capture.lidar_path.resolve())])
    if cfg.capture.calibration_path is not None:
        args.extend(["--calibration", str(cfg.capture.calibration_path.resolve())])
    return args


def _run_nvblox(
    cfg: SceneConfig,
    *,
    manifest_path: Path,
    output_mesh: Path,
) -> tuple[Path, str]:
    tool = external_tool(cfg, "nvblox", default="nvblox")
    assert tool is not None
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    tool.run(*nvblox_args(cfg, manifest_path=manifest_path, output_mesh=output_mesh))
    if not output_mesh.is_file():
        raise FileNotFoundError(f"nvblox did not create {output_mesh}")
    return output_mesh, FRAME_USD


def openmvs_commands(
    cfg: SceneConfig,
    *,
    work_dir: Path,
    dense_dir: Path,
) -> list[tuple[ExternalToolAdapter, list[str]]]:
    interface = external_tool(cfg, "openmvs_interface", default="InterfaceCOLMAP")
    densify = external_tool(cfg, "openmvs_densify", default="DensifyPointCloud")
    reconstruct = external_tool(cfg, "openmvs_reconstruct", default="ReconstructMesh")
    refine = external_tool(cfg, "openmvs_refine", default="RefineMesh")
    assert interface and densify and reconstruct and refine
    scene = work_dir / "scene.mvs"
    dense = work_dir / "scene_dense.mvs"
    mesh = work_dir / "scene_dense_mesh.mvs"
    refined = work_dir / "scene_dense_mesh_refine.mvs"
    return [
        (
            interface,
            [
                "-i",
                str(dense_dir.resolve()),
                "-o",
                str(scene),
                "--image-folder",
                str((dense_dir / "images").resolve()),
            ],
        ),
        (densify, [str(scene), "-o", str(dense)]),
        (reconstruct, [str(dense), "-o", str(mesh)]),
        (refine, [str(mesh), "-o", str(refined)]),
    ]


def _run_openmvs(cfg: SceneConfig, *, work_dir: Path) -> tuple[Path, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    dense_dir = work_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    colmap = ExternalToolAdapter("colmap", [resolve_colmap(cfg)])
    colmap.run(
        "image_undistorter",
        "--image_path",
        str((cfg.nerfstudio_data_dir / "images").resolve()),
        "--input_path",
        str((cfg.nerfstudio_data_dir / "colmap" / "sparse" / "0").resolve()),
        "--output_path",
        str(dense_dir.resolve()),
        "--output_type",
        "COLMAP",
    )
    # Proxy OpenMVS wrappers fall back to sparse points when dense fusion is missing.
    # Always point at this scene's COLMAP TXT (not a hardcoded shared workspace path).
    points_txt = (cfg.colmap_txt_dir / "points3D.txt").resolve()
    if not points_txt.is_file():
        raise FileNotFoundError(
            f"Missing {points_txt}; run reconstruct (COLMAP TXT export) before static geometry"
        )
    proxy_env = {"SCAN2USD_COLMAP_TXT": str(points_txt)}
    for tool, args in openmvs_commands(cfg, work_dir=work_dir, dense_dir=dense_dir):
        tool.cwd = work_dir
        tool.env.update(proxy_env)
        tool.run(*args)
    preferred = (
        work_dir / "scene_dense_mesh_refine.ply",
        work_dir / "scene_dense_mesh.ply",
        work_dir / "scene_dense_mesh_refine.obj",
        work_dir / "scene_dense_mesh.obj",
    )
    source = next((path for path in preferred if path.is_file()), None)
    if source is None:
        candidates = sorted(work_dir.glob("*.ply")) + sorted(work_dir.glob("*.obj"))
        source = candidates[-1] if candidates else None
    if source is None:
        raise FileNotFoundError(f"OpenMVS completed without a mesh under {work_dir}")
    return source, FRAME_COLMAP


def build_static_scene(
    cfg: SceneConfig,
    manifest: SceneManifest,
    *,
    manifest_path: Path,
) -> dict[str, Path]:
    """Run the modality-specific dense backend and emit canonical collision/proxy meshes."""
    require_metric_geometry(cfg, manifest)
    build_root = cfg.workspace_dir / "build" / "geometry"
    if cfg.capture.modality in {"rgbd", "lidar"}:
        raw_mesh, source_frame = _run_nvblox(
            cfg,
            manifest_path=manifest_path,
            output_mesh=build_root / "nvblox" / "static_raw.ply",
        )
    else:
        raw_mesh, source_frame = _run_openmvs(cfg, work_dir=build_root / "openmvs")

    transform = source_to_usd_transform(manifest, source_frame)
    collision = build_root / "static_collision.ply"
    collision_report = process_mesh_file(
        raw_mesh,
        collision,
        source_to_usd=transform,
        target_faces=cfg.geometry.target_static_triangles,
        report_path=build_root / "static_collision_report.json",
    )
    collision_mesh = clean_mesh(load_mesh(collision))
    proxy_mesh = simplify_mesh(
        collision_mesh,
        max(10_000, cfg.geometry.target_static_triangles // 4),
    )
    proxy = build_root / "static_proxy.ply"
    proxy_mesh.export(proxy)
    chunks = split_static_mesh(collision_mesh, build_root / "chunks")

    manifest.register_artifact(
        artifact_id="static_collision_mesh",
        kind="triangle_collision_mesh",
        path=collision,
        producer="nvblox" if cfg.capture.modality in {"rgbd", "lidar"} else "openmvs",
        metadata={
            "source_frame": source_frame,
            "static": True,
            "collision_approximation": "none",
            "report": collision_report.__dict__,
        },
    )
    manifest.register_artifact(
        artifact_id="static_render_proxy",
        kind="render_proxy_mesh",
        path=proxy,
        producer="scan2usd.mesh_ops",
        metadata={"report": mesh_report(proxy_mesh, proxy).__dict__},
    )
    for name, path in chunks.items():
        manifest.register_artifact(
            artifact_id=f"static_chunk_{name}",
            kind="static_mesh_chunk",
            path=path,
            producer="scan2usd.mesh_ops",
            metadata={"chunk": name},
        )
    report = {
        "backend": "nvblox" if cfg.capture.modality in {"rgbd", "lidar"} else "openmvs",
        "raw_mesh": str(raw_mesh.resolve()),
        "source_frame": source_frame,
        "collision": str(collision.resolve()),
        "proxy": str(proxy.resolve()),
        "chunks": {name: str(path.resolve()) for name, path in chunks.items()},
    }
    depth_validation = raw_mesh.parent / "depth_validation.json"
    if depth_validation.is_file():
        report["depth_validation"] = json.loads(depth_validation.read_text(encoding="utf-8"))
        collision_artifact = manifest.artifact("static_collision_mesh")
        if collision_artifact is not None:
            collision_artifact.metadata["depth_validation"] = report["depth_validation"]
    (build_root / "static_scene_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"collision": collision, "proxy": proxy, **chunks}
