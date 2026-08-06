"""Headless Isaac Sim RGB rendering of the packaged scene at held-out capture cameras.

Runs under Isaac Sim's Python (``external.isaac_python``); deliberately self-contained
(no scan2usd import). Camera poses and intrinsics are read from the COLMAP TXT model —
the same raw COLMAP frame the ParticleField is exported in — and mapped into USD world
with the audited COLMAP→USD matrix from the scene manifest. Never use the Nerfstudio
``transforms.json`` poses here: they live in a different (applied_transform) frame.

Output: ``<output>/<frame_stem>.png`` per held-out view + ``render_report.json``.
These renders feed ``scan2usd validate-usd`` / the photorealism metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp  # noqa: E402


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float):
    import numpy as np

    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _parse_colmap_txt(colmap_txt: Path) -> tuple[dict, dict]:
    """Return ({image_name: c2w_opencv_4x4}, {camera_id: intrinsics dict})."""
    import numpy as np

    cameras: dict[int, dict] = {}
    for line in (colmap_txt / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cam_id, model = int(parts[0]), parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model in {"OPENCV", "PINHOLE", "OPENCV_FISHEYE", "RADIAL", "SIMPLE_RADIAL"}:
            if model in {"RADIAL", "SIMPLE_RADIAL"}:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise SystemExit(f"Unsupported COLMAP camera model: {model}")
        cameras[cam_id] = {
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    poses: dict[str, dict] = {}
    lines = (colmap_txt / "images.txt").read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    # images.txt alternates: pose line, 2D-points line.
    for pose_line in data_lines[::2]:
        parts = pose_line.split()
        qw, qx, qy, qz = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        r_w2c = _quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = r_w2c.T
        c2w[:3, 3] = -(r_w2c.T @ t)
        poses[name] = {"c2w_opencv": c2w, "camera_id": cam_id}
    return poses, cameras


def _colmap_to_usd_from_manifest(manifest_path: Path):
    import numpy as np

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for transform in raw.get("transforms", []):
        if (
            transform.get("source_frame") == "colmap_world"
            and transform.get("target_frame") == "usd_world_z_up_meters"
        ):
            return np.asarray(transform["matrix"], dtype=np.float64)
    raise SystemExit(f"No colmap_world→usd_world transform in {manifest_path}")


def _usd_camera_pose(c2w_opencv, colmap_to_usd):
    """USD camera-to-world (column-vector math): T @ c2w_cv @ flip, scale stripped."""
    import numpy as np

    flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV (+Z fwd, +Y down) → OpenGL/USD
    m = colmap_to_usd @ c2w_opencv @ flip
    rotation = m[:3, :3].copy()
    for col in range(3):
        norm = float(np.linalg.norm(rotation[:, col]))
        if norm < 1e-12:
            raise SystemExit("Degenerate camera rotation after transform")
        rotation[:, col] /= norm
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = m[:3, 3]
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True, help="held_out.json")
    parser.add_argument("--colmap-txt", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="scene_manifest.json with the COLMAP→USD transform (default: next to stage)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=48)
    parser.add_argument("--settle-frames", type=int, default=2, help="app.update() per view")
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=16,
        help="RTX subframes accumulated per capture (higher = less noise, slower)",
    )
    parser.add_argument("--width", type=int, default=0, help="Override render width")
    parser.add_argument("--height", type=int, default=0, help="Override render height")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1920,
        help=(
            "Cap the render's longest side. Capture intrinsics can be 4K, and a "
            "3840x2160 render product silently yields all-black frames on an 8 GB "
            "card. Metrics resize to the reference anyway. 0 disables the cap."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render only the first N held-out views (0 = all). Useful for smoke tests.",
    )
    args = parser.parse_args()

    stage_path = args.stage.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else stage_path.parent / "scene_manifest.json"
    )
    held_out = json.loads(args.held_out.read_text(encoding="utf-8"))
    names = [str(item["file"]) for item in held_out.get("images", [])]
    if not names:
        raise SystemExit(f"No held-out images listed in {args.held_out}")
    if args.limit > 0:
        names = names[: args.limit]

    import numpy as np

    poses, cameras = _parse_colmap_txt(args.colmap_txt.expanduser().resolve())
    colmap_to_usd = _colmap_to_usd_from_manifest(manifest_path)

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
        }
    )
    report: dict = {
        "stage": str(stage_path),
        "manifest": str(manifest_path),
        "colmap_txt": str(args.colmap_txt.resolve()),
        "views": [],
        "errors": [],
    }
    try:
        import carb
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom

        settings = carb.settings.get_settings()
        # Deterministic exposure: the metric compares against fixed-exposure video
        # frames, so RTX auto-exposure must not adapt per view.
        settings.set("/rtx/post/histogram/enabled", False)

        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"Could not open stage {stage_path}")
        stage = context.get_stage()
        stage.SetLoadRules(Usd.StageLoadRules.LoadAll())
        stage.Load()
        for _ in range(args.warmup_frames):
            app.update()

        camera_path = "/World/HeldOutCamera"
        usd_camera = UsdGeom.Camera.Define(stage, camera_path)
        usd_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
        xform = UsdGeom.Xformable(usd_camera.GetPrim())
        transform_op = xform.AddTransformOp()

        first = poses.get(names[0])
        if first is None:
            raise RuntimeError(f"{names[0]} not found in COLMAP images.txt")
        intrinsics = cameras[first["camera_id"]]
        width = args.width or intrinsics["width"]
        height = args.height or intrinsics["height"]
        if args.max_dim and max(width, height) > args.max_dim:
            shrink = args.max_dim / float(max(width, height))
            width = max(1, int(round(width * shrink)))
            height = max(1, int(round(height * shrink)))
            print(
                f"Render capped to {width}x{height} "
                f"(capture intrinsics are {intrinsics['width']}x{intrinsics['height']}).",
                flush=True,
            )
        focal = 10.0
        h_aperture = focal * intrinsics["width"] / intrinsics["fx"]
        v_aperture = focal * intrinsics["height"] / intrinsics["fy"]
        usd_camera.CreateFocalLengthAttr(focal)
        usd_camera.CreateHorizontalApertureAttr(h_aperture)
        usd_camera.CreateVerticalApertureAttr(v_aperture)
        usd_camera.CreateHorizontalApertureOffsetAttr(
            (intrinsics["cx"] - intrinsics["width"] / 2.0) / intrinsics["fx"] * focal
        )
        usd_camera.CreateVerticalApertureOffsetAttr(
            (intrinsics["height"] / 2.0 - intrinsics["cy"]) / intrinsics["fy"] * focal
        )

        render_product = rep.create.render_product(camera_path, (int(width), int(height)))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)

        args.output.mkdir(parents=True, exist_ok=True)
        for name in names:
            entry = poses.get(name)
            if entry is None:
                report["errors"].append(f"{name}: not in COLMAP model")
                continue
            pose = _usd_camera_pose(entry["c2w_opencv"], colmap_to_usd)
            # USD Gf.Matrix4d uses row-vector convention: author the transpose.
            transform_op.Set(Gf.Matrix4d(*np.asarray(pose.T, dtype=np.float64).flatten()))
            for _ in range(args.settle_frames):
                app.update()
            # app.update() alone never fills the annotator: only a synchronous
            # orchestrator step schedules a capture for attached annotators.
            rep.orchestrator.step(
                rt_subframes=max(1, args.rt_subframes),
                pause_timeline=True,
                wait_for_render=True,
            )
            data = annotator.get_data()
            if isinstance(data, dict):
                data = data.get("data", data)
            image = np.asarray(data)
            if image.ndim != 3 or image.shape[0] == 0:
                report["errors"].append(
                    f"{name}: annotator returned no image (shape={getattr(image, 'shape', None)})"
                )
                continue
            # A render product the GPU could not service comes back as a valid
            # array of zeros rather than an error. Scoring those produced a
            # confident-looking 3.79/100 for a scene that renders fine at a
            # smaller size, so refuse to write a frame with no image in it.
            if float(np.asarray(image[..., :3]).std()) < 1e-6:
                report["errors"].append(
                    f"{name}: render is uniform (mean="
                    f"{float(np.asarray(image[..., :3]).mean()):.1f}) — the render "
                    "product returned an empty buffer; try a smaller --max-dim"
                )
                continue
            out_path = args.output / f"{Path(name).stem}.png"
            Image.fromarray(image[..., :3].astype(np.uint8)).save(out_path)
            report["views"].append(
                {
                    "image": name,
                    "render": str(out_path.resolve()),
                    "camera_to_world_usd": pose.tolist(),
                }
            )
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        report["rendered"] = len(report["views"])
        report["expected"] = len(names)
        report["passed"] = bool(report["views"]) and not report["errors"]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "render_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        app.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
