"""COLMAP dense/poisson stand-in for OpenMVS CLI tools when OpenMVS is not installed."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from scan2usd.reconstruction.colmap_io import parse_points3d_txt


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interface_colmap(args: argparse.Namespace) -> None:
    dense_dir = args.input.resolve()
    if not (dense_dir / "sparse").is_dir():
        raise FileNotFoundError(f"COLMAP undistorted sparse model missing under {dense_dir}")
    _write_state(
        args.output,
        {
            "kind": "colmap_dense_proxy",
            "dense_dir": str(dense_dir),
            "image_folder": str((dense_dir / args.image_folder).resolve()),
        },
    )


def densify_point_cloud(args: argparse.Namespace) -> None:
    state = _read_state(args.input)
    dense_dir = Path(state["dense_dir"])
    fused = dense_dir / "fused.ply"
    if fused.is_file():
        state["fused_point_cloud"] = str(fused.resolve())
        _write_state(args.output, state)
        return
    _write_state(args.output, state)


def reconstruct_mesh(args: argparse.Namespace) -> None:
    state = _read_state(args.input)
    fused = Path(state.get("fused_point_cloud", ""))
    work_dir = args.output.parent
    if fused.is_file():
        mesh = fused.with_name("meshed.ply")
        subprocess.run(
            [
                shutil.which("colmap") or "colmap",
                "poisson_mesher",
                "--input_path",
                str(fused),
                "--output_path",
                str(mesh),
            ],
            check=True,
        )
        state["mesh"] = str(mesh.resolve())
    else:
        import os

        sparse_txt = os.environ.get("SCAN2USD_COLMAP_TXT", "")
        if not sparse_txt:
            raise RuntimeError(
                "Dense fusion unavailable; set SCAN2USD_COLMAP_TXT to sparse points3D.txt "
                "for preview meshing"
            )
        sparse_fallback_mesh(
            argparse.Namespace(
                sparse_txt=Path(sparse_txt),
                work_dir=work_dir,
            )
        )
        state["mesh"] = str((work_dir / "scene_dense_mesh.ply").resolve())
    _write_state(args.output, state)


def refine_mesh(args: argparse.Namespace) -> None:
    state = _read_state(args.input)
    mesh = Path(state["mesh"])
    if not mesh.is_file():
        raise FileNotFoundError(f"Mesh missing: {mesh}")
    refined = args.output.with_suffix(".ply")
    if mesh.resolve() != refined.resolve():
        shutil.copy2(mesh, refined)
    state["refined_mesh"] = str(refined.resolve())
    _write_state(args.output, state)


def sparse_fallback_mesh(args: argparse.Namespace) -> None:
    """Fast preview path: mesh sparse COLMAP points with trimesh."""
    import numpy as np
    import trimesh

    sparse_txt = args.sparse_txt.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _ids, points = parse_points3d_txt(sparse_txt)
    if len(points) == 0:
        raise RuntimeError(f"No COLMAP points in {sparse_txt}")
    refined = work_dir / "scene_dense_mesh.ply"
    cloud = trimesh.PointCloud(points)
    try:
        mesh = cloud.convex_hull
    except Exception:
        mesh = trimesh.Trimesh(vertices=points, faces=np.empty((0, 3), dtype=np.int64))
    mesh.export(refined)
    _write_state(
        work_dir / "scene_dense_mesh.mvs",
        {"mesh": str(refined.resolve()), "source": "sparse_fallback"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    interface = sub.add_parser("interface-colmap")
    interface.add_argument("-i", "--input", type=Path, required=True)
    interface.add_argument("-o", "--output", type=Path, required=True)
    interface.add_argument("--image-folder", default="images")
    interface.set_defaults(func=interface_colmap)

    densify = sub.add_parser("densify")
    densify.add_argument("input", type=Path)
    densify.add_argument("-o", "--output", type=Path, required=True)
    densify.set_defaults(func=densify_point_cloud)

    reconstruct = sub.add_parser("reconstruct")
    reconstruct.add_argument("input", type=Path)
    reconstruct.add_argument("-o", "--output", type=Path, required=True)
    reconstruct.set_defaults(func=reconstruct_mesh)

    refine = sub.add_parser("refine")
    refine.add_argument("input", type=Path)
    refine.add_argument("-o", "--output", type=Path, required=True)
    refine.set_defaults(func=refine_mesh)

    sparse = sub.add_parser("sparse-fallback")
    sparse.add_argument("--sparse-txt", type=Path, required=True)
    sparse.add_argument("--work-dir", type=Path, required=True)
    sparse.set_defaults(func=sparse_fallback_mesh)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
