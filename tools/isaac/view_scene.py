"""Open a Scan2USD stage in Isaac Sim with payloads loaded and camera framed."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


def _add_z0_ground_marker(stage) -> str | None:
    """Insert a thin, non-colliding grid at Z=0 for visual floor checks."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path = "/World/Debug/GroundZ0"
    if stage.GetPrimAtPath(path):
        return path

    UsdGeom.Xform.Define(stage, "/World/Debug")
    grid = UsdGeom.Mesh.Define(stage, path)
    # 4 m square, centered on origin, slightly above Z=0 to avoid z-fighting.
    half = 2.0
    z = 0.002
    grid.CreatePointsAttr(
        [
            Gf.Vec3f(-half, -half, z),
            Gf.Vec3f(half, -half, z),
            Gf.Vec3f(half, half, z),
            Gf.Vec3f(-half, half, z),
        ]
    )
    grid.CreateFaceVertexCountsAttr([4])
    grid.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    grid.CreateExtentAttr([Gf.Vec3f(-half, -half, z), Gf.Vec3f(half, half, z)])
    grid.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.75, 0.35)])
    grid.CreateDisplayOpacityAttr([0.35])
    grid.GetPrim().CreateAttribute(
        "scan2usd:debugGroundMarker",
        Sdf.ValueTypeNames.Bool,
        custom=True,
    ).Set(True)

    # Unlit look so the marker stays readable under any lighting.
    material_path = "/World/Debug/GroundZ0Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.35, 0.75, 0.35)
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(grid).Bind(material)
    return path


def _parse_colmap_pose(images_txt: Path, index_fraction: float = 0.5):
    """Camera-to-world (OpenCV) of a real capture pose partway along the trajectory."""
    import numpy as np

    lines = [
        ln
        for ln in images_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    poses = lines[::2]  # images.txt alternates pose / 2D-points lines
    if not poses:
        return None
    parts = poses[min(len(poses) - 1, int(len(poses) * index_fraction))].split()
    qw, qx, qy, qz = (float(v) for v in parts[1:5])
    t = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    r_w2c = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    c2w = np.eye(4)
    c2w[:3, :3] = r_w2c.T
    c2w[:3, 3] = -(r_w2c.T @ t)
    return c2w


def _place_camera_inside(stage, stage_path: Path, args) -> bool:
    """
    Start the viewport at a pose the capture actually visited.

    Framing the whole scene puts the camera outside the room, which for an
    inside-out capture is the one place a Gaussian model has no training data —
    you get haze instead of the scene. Reusing a real capture pose guarantees a
    viewpoint the reconstruction is valid for. Falls back to framing when the
    COLMAP model or transform is unavailable.
    """
    import json

    import numpy as np
    from pxr import Gf, UsdGeom

    workspace = stage_path.parent.parent
    colmap_txt = args.colmap_txt or (workspace / "colmap_txt" / "images.txt")
    manifest = args.manifest or (workspace / "scene_manifest.json")
    if not (Path(colmap_txt).is_file() and Path(manifest).is_file()):
        print("[scan2usd] no COLMAP/manifest next to the stage; framing instead", flush=True)
        return False

    c2w = _parse_colmap_pose(Path(colmap_txt), args.along_path)
    if c2w is None:
        return False
    raw = json.loads(Path(manifest).read_text(encoding="utf-8"))
    matrix = next(
        (
            np.asarray(t["matrix"], dtype=np.float64)
            for t in raw.get("transforms", [])
            if t.get("source_frame") == "colmap_world"
            and t.get("target_frame") == "usd_world_z_up_meters"
        ),
        None,
    )
    if matrix is None:
        print("[scan2usd] no COLMAP->USD transform in manifest; framing instead", flush=True)
        return False

    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV -> USD camera convention
    pose = matrix @ c2w @ flip
    rotation = pose[:3, :3].copy()
    for col in range(3):
        norm = float(np.linalg.norm(rotation[:, col]))
        if norm < 1e-12:
            return False
        rotation[:, col] /= norm
    world = np.eye(4)
    world[:3, :3] = rotation
    world[:3, 3] = pose[:3, 3]

    camera_path = "/World/InsideCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
    camera.CreateFocalLengthAttr(float(args.focal_mm))
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    # USD Gf matrices are row-vector, so author the transpose.
    xform.AddTransformOp().Set(Gf.Matrix4d(*world.T.flatten()))

    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        viewport.camera_path = camera_path
    except Exception as exc:  # noqa: BLE001
        print(f"[scan2usd] could not switch viewport camera: {exc}", flush=True)
        return False
    print(
        f"[scan2usd] viewport starts at a real capture pose "
        f"({args.along_path:.0%} along the path) at {np.round(world[:3, 3], 2)}",
        flush=True,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=Path,
        default=Path("/home/arpit/Scan2USD/workspace/usd/scene.usd"),
    )
    parser.add_argument(
        "--ground-marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a non-colliding translucent plane at Z=0 (Isaac ground)",
    )
    parser.add_argument(
        "--start",
        choices=["inside", "framed"],
        default="inside",
        help="Where the viewport camera starts. 'inside' reuses a real capture "
        "pose (the reconstruction is only valid where the camera actually went); "
        "'framed' fits the whole scene, which for a room means viewing it from "
        "outside where a splat has no data.",
    )
    parser.add_argument(
        "--along-path",
        type=float,
        default=0.5,
        help="How far along the capture trajectory to start (0=first frame, 1=last).",
    )
    parser.add_argument("--focal-mm", type=float, default=18.0, help="Viewport focal length")
    parser.add_argument("--colmap-txt", type=Path, default=None, help="images.txt override")
    parser.add_argument("--manifest", type=Path, default=None, help="scene_manifest.json override")
    args = parser.parse_args()
    stage_path = args.stage.expanduser().resolve()
    if not stage_path.is_file():
        raise SystemExit(f"USD stage not found: {stage_path}")

    app = SimulationApp(
        {
            "headless": False,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
            "width": 1920,
            "height": 1080,
        }
    )
    try:
        import omni.kit.commands
        import omni.usd
        from pxr import Usd, UsdGeom

        context = omni.usd.get_context()
        print(f"[scan2usd] Opening {stage_path}", flush=True)
        ok = context.open_stage(str(stage_path))
        for _ in range(60):
            app.update()

        stage = context.get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open stage (ok={ok}): {stage_path}")

        # ParticleField splat is authored as a payload; load it explicitly.
        stage.SetLoadRules(Usd.StageLoadRules.LoadAll())
        stage.Load()
        for _ in range(60):
            app.update()

        if args.ground_marker:
            marker = _add_z0_ground_marker(stage)
            print(f"[scan2usd] Z=0 ground marker at {marker}", flush=True)

        default_prim = stage.GetDefaultPrim()
        print(
            f"[scan2usd] defaultPrim={default_prim.GetPath() if default_prim else None}",
            flush=True,
        )
        particle_fields = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if str(prim.GetTypeName()).startswith("ParticleField")
        ]
        meshes = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.IsA(UsdGeom.Mesh)
        ]
        print(f"[scan2usd] ParticleFields={particle_fields}", flush=True)
        print(f"[scan2usd] MeshCount={len(meshes)}", flush=True)

        placed_inside = False
        if args.start == "inside":
            placed_inside = _place_camera_inside(stage, stage_path, args)
        if not placed_inside:
            try:
                omni.kit.commands.execute(
                    "FramePrimsCommand",
                    prim_paths=["/World"],
                    use_bbox_cache=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[scan2usd] FramePrimsCommand skipped: {exc}", flush=True)

        print("[scan2usd] Scene loaded. Close the Isaac Sim window to exit.", flush=True)
        while app.is_running():
            app.update()
    finally:
        app.close()


if __name__ == "__main__":
    main()
