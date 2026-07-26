"""Headless Isaac Sim structural, collision, and rigid-body smoke validation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp


def _xyz(matrix):
    value = matrix.ExtractTranslation()
    return [float(value[0]), float(value[1]), float(value[2])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=120)
    args = parser.parse_args()

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
        }
    )
    report = {
        "stage": str(args.stage.resolve()),
        "checks": {},
        "errors": [],
        "warnings": [],
    }
    try:
        import carb
        import omni.timeline
        import omni.usd
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        context = omni.usd.get_context()
        context.open_stage(str(args.stage.resolve()))
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim could not open the USD stage")

        default_prim = stage.GetDefaultPrim()
        report["checks"]["stage_metadata"] = {
            "passed": bool(default_prim)
            and abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) < 1e-9
            and UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z,
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        }

        particle_fields = []
        meshes = []
        collisions = []
        rigid_bodies = []
        unresolved_assets = []
        for prim in stage.Traverse():
            type_name = str(prim.GetTypeName())
            if type_name.startswith("ParticleField"):
                particle_fields.append(str(prim.GetPath()))
            if prim.IsA(UsdGeom.Mesh):
                meshes.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collisions.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(str(prim.GetPath()))
            for attribute in prim.GetAttributes():
                value = attribute.Get()
                if hasattr(value, "path") and value.path and not value.resolvedPath:
                    unresolved_assets.append(f"{prim.GetPath()}:{attribute.GetName()}={value.path}")

        report["checks"]["composition"] = {
            "passed": not unresolved_assets,
            "particle_fields": particle_fields,
            "mesh_count": len(meshes),
            "collision_count": len(collisions),
            "rigid_bodies": rigid_bodies,
            "unresolved_assets": unresolved_assets,
        }
        report["checks"]["particle_field"] = {
            "passed": len(particle_fields) >= 1,
            "count": len(particle_fields),
        }
        report["checks"]["collision_schema"] = {
            "passed": len(collisions) >= 1,
            "count": len(collisions),
        }

        timeline = omni.timeline.get_timeline_interface()
        initial = {
            path: _xyz(omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path)))
            for path in rigid_bodies
        }
        timeline.play()
        for _ in range(args.settle_steps):
            app.update()
        settled = {
            path: _xyz(omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path)))
            for path in rigid_bodies
        }
        finite = all(math.isfinite(v) for point in settled.values() for v in point)
        report["checks"]["drop_rest"] = {
            "passed": finite,
            "initial": initial,
            "settled": settled,
        }

        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        ).ComputeWorldBound(default_prim).ComputeAlignedRange()
        center = bbox.GetMidpoint()
        ray_origin = carb.Float3(float(center[0]), float(center[1]), float(bbox.GetMax()[2] + 2.0))
        ray_direction = carb.Float3(0.0, 0.0, -1.0)
        ray = get_physx_scene_query_interface().raycast_closest(
            ray_origin,
            ray_direction,
            float(max(5.0, bbox.GetSize()[2] + 4.0)),
        )
        report["checks"]["static_raycast"] = {
            "passed": bool(ray.get("hit")),
            "hit": str(ray.get("rigidBody", "")),
            "distance": float(ray.get("distance", 0.0)),
        }

        if rigid_bodies:
            selected = rigid_bodies[0]
            prim = stage.GetPrimAtPath(selected)
            before_push = _xyz(omni.usd.get_world_transform_matrix(prim))
            rigid_api = UsdPhysics.RigidBodyAPI(prim)
            rigid_api.CreateVelocityAttr().Set(Gf.Vec3f(0.25, 0.0, 0.0))
            for _ in range(45):
                app.update()
            after_push = _xyz(omni.usd.get_world_transform_matrix(prim))
            push_distance = math.dist(before_push, after_push)
            report["checks"]["push"] = {
                "passed": push_distance > 1e-4,
                "object": selected,
                "distance_m": push_distance,
            }

            timeline.pause()
            rigid_api.CreateKinematicEnabledAttr().Set(True)
            transform_attr = prim.GetAttribute("xformOp:transform")
            carried = False
            if transform_attr and transform_attr.HasAuthoredValueOpinion():
                matrix = transform_attr.Get()
                current = _xyz(omni.usd.get_world_transform_matrix(prim))
                target = Gf.Vec3d(current[0] + 0.25, current[1], current[2] + 0.25)
                matrix.SetTranslateOnly(target)
                transform_attr.Set(matrix)
                for _ in range(4):
                    app.update()
                carried_pos = _xyz(omni.usd.get_world_transform_matrix(prim))
                carried = math.dist(current, carried_pos) > 0.1
            rigid_api.GetKinematicEnabledAttr().Set(False)
            timeline.play()
            for _ in range(60):
                app.update()
            placed = _xyz(omni.usd.get_world_transform_matrix(prim))
            report["checks"]["pick_carry_place"] = {
                "passed": carried and all(math.isfinite(value) for value in placed),
                "object": selected,
                "placed": placed,
            }
        else:
            report["checks"]["push"] = {"passed": True, "skipped": "no rigid objects"}
            report["checks"]["pick_carry_place"] = {
                "passed": True,
                "skipped": "no rigid objects",
            }
        timeline.stop()
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        passed = not report["errors"] and all(
            check.get("passed", False) for check in report["checks"].values()
        )
        report["passed"] = passed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        app.close()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
