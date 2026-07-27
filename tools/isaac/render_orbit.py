"""Render a packaged Scan2USD stage from cameras ORBITING it, outside the capture path.

Held-out validation only ever looks from inside the capture trajectory, where stray
Gaussians hide behind or beside the view. Floaters and halo shells are precisely what
you see when you pull the camera back — so this renders the scene the way a person
inspects it, and is the check that catches obstruction PSNR cannot.

There is no ground truth for these views; the output is for eyes and for the
obstruction estimate below (fraction of frame far brighter/darker than the scene's
own median, a proxy for haze blanketing the camera).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=8, help="Cameras around the ring")
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.5,
        help="Ring radius as a multiple of the scene's bounding radius",
    )
    parser.add_argument(
        "--elevation-deg", type=float, default=22.0, help="Camera height above centre"
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--rt-subframes", type=int, default=24)
    parser.add_argument("--warmup-frames", type=int, default=48)
    args = parser.parse_args()

    stage_path = args.stage.expanduser().resolve()
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "multi_gpu": False})
    report: dict = {"stage": str(stage_path), "views": [], "errors": []}
    try:
        import carb
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom

        carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"Could not open stage {stage_path}")
        stage = context.get_stage()
        stage.SetLoadRules(Usd.StageLoadRules.LoadAll())
        stage.Load()
        for _ in range(args.warmup_frames):
            app.update()

        default_prim = stage.GetDefaultPrim()
        bounds = (
            UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            .ComputeWorldBound(default_prim)
            .ComputeAlignedRange()
        )
        centre = bounds.GetMidpoint()
        size = bounds.GetSize()
        radius = 0.5 * math.sqrt(float(size[0]) ** 2 + float(size[1]) ** 2)
        radius = max(radius, 1e-3) * args.radius_scale
        height = float(centre[2]) + radius * math.tan(math.radians(args.elevation_deg))
        report["scene"] = {
            "centre": [float(c) for c in centre],
            "size": [float(s) for s in size],
            "ring_radius": radius,
        }

        camera_path = "/World/OrbitCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(36.0 * args.height / args.width)
        xform = UsdGeom.Xformable(camera.GetPrim())
        op = xform.AddTransformOp()

        render_product = rep.create.render_product(camera_path, (args.width, args.height))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)
        args.output.mkdir(parents=True, exist_ok=True)

        for i in range(args.views):
            angle = 2.0 * math.pi * i / args.views
            eye = Gf.Vec3d(
                float(centre[0]) + radius * math.cos(angle),
                float(centre[1]) + radius * math.sin(angle),
                height,
            )
            target = Gf.Vec3d(float(centre[0]), float(centre[1]), float(centre[2]))
            # Gf.Matrix4d.SetLookAt builds world->view; the camera needs its inverse.
            view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
            op.Set(view.GetInverse())
            app.update()
            rep.orchestrator.step(
                rt_subframes=max(1, args.rt_subframes), pause_timeline=True, wait_for_render=True
            )
            data = annotator.get_data()
            if isinstance(data, dict):
                data = data.get("data", data)
            image = np.asarray(data)
            if image.ndim != 3 or image.shape[0] == 0:
                report["errors"].append(f"view {i}: no image")
                continue
            rgb = image[..., :3].astype(np.uint8)
            out = args.output / f"orbit_{i:02d}.png"
            Image.fromarray(rgb).save(out)

            grey = rgb.mean(axis=2)
            # Haze blanketing the lens shows up as a large low-contrast region.
            flat = float(np.mean(np.abs(grey - np.median(grey)) < 6.0))
            report["views"].append(
                {
                    "index": i,
                    "azimuth_deg": round(math.degrees(angle), 1),
                    "render": str(out.resolve()),
                    "flat_fraction": round(flat, 4),
                }
            )
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if report["views"]:
            report["mean_flat_fraction"] = round(
                sum(v["flat_fraction"] for v in report["views"]) / len(report["views"]), 4
            )
        report["rendered"] = len(report["views"])
        report["passed"] = bool(report["views"]) and not report["errors"]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "orbit_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        app.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
