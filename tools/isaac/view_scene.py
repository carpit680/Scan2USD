"""Open a Scan2USD stage in Isaac Sim with payloads loaded and camera framed."""

from __future__ import annotations

import argparse
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
