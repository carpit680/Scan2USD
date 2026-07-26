#!/usr/bin/env python3
"""Masked COLMAP sparse → textured mesh object reconstructor for Scan2USD preview.

CLI contract (docs/USAGE.md):
  --images --masks --colmap --instance-id --output-mesh --texture-resolution
  [--detail-capture]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from scan2usd.reconstruction.colmap_io import (
    export_colmap_to_txt,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
    project_points,
)


def _parse_points3d_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[int] = []
    xyz: list[list[float]] = []
    rgb: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        ids.append(int(parts[0]))
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not ids:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(xyz, dtype=np.float64),
        np.asarray(rgb, dtype=np.uint8),
    )


def _mask_path_for_image(mask_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for candidate in (
        mask_dir / f"{stem}.png",
        mask_dir / image_name,
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{Path(image_name).name}",
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_mask(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as raw:
        mask = np.asarray(
            raw.convert("L").resize((width, height), Image.Resampling.NEAREST)
        )
    return mask > 127


def collect_masked_points(
    *,
    images_dir: Path,
    masks_dir: Path,
    colmap_dir: Path,
    min_votes: int = 2,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """Return (xyz, rgb, best_mask_image) in COLMAP world coordinates."""
    with tempfile.TemporaryDirectory(prefix="scan2usd_obj_recon_") as tmp:
        txt_dir = Path(tmp)
        # Accept either binary model dir or already-text model dir.
        if (colmap_dir / "points3D.bin").is_file() or (colmap_dir / "cameras.bin").is_file():
            export_colmap_to_txt(colmap_dir, txt_dir)
        else:
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                src = colmap_dir / name
                if not src.is_file():
                    raise FileNotFoundError(f"Missing COLMAP model file: {src}")
                (txt_dir / name).write_bytes(src.read_bytes())

        cameras = parse_cameras_txt(txt_dir / "cameras.txt")
        images = parse_images_txt(txt_dir / "images.txt")
        _ids, xyz = parse_points3d_txt(txt_dir / "points3D.txt")
        _ids2, _xyz2, rgb = _parse_points3d_rgb(txt_dir / "points3D.txt")
        if xyz.shape[0] == 0:
            raise RuntimeError("COLMAP sparse model has no 3D points")

        votes = np.zeros((xyz.shape[0],), dtype=np.int32)
        color_accum = np.zeros((xyz.shape[0], 3), dtype=np.float64)
        color_count = np.zeros((xyz.shape[0],), dtype=np.int32)
        best_view: Path | None = None
        best_area = 0

        for image_name, pose in images.items():
            mask_path = _mask_path_for_image(masks_dir, image_name)
            if mask_path is None:
                continue
            intr = cameras[pose.camera_id]
            mask = _load_mask(mask_path, intr.width, intr.height)
            area = int(mask.sum())

            image_path = images_dir / image_name
            if not image_path.is_file():
                stem = Path(image_name).stem
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = images_dir / f"{stem}{ext}"
                    if candidate.is_file():
                        image_path = candidate
                        break

            if area > best_area and image_path.is_file():
                best_area = area
                best_view = image_path

            uv, depth = project_points(xyz, pose.qvec, pose.tvec, intr)
            u = np.round(uv[:, 0]).astype(np.int32)
            v = np.round(uv[:, 1]).astype(np.int32)
            valid = (
                (depth > 0.05)
                & (u >= 0)
                & (v >= 0)
                & (u < intr.width)
                & (v < intr.height)
            )
            if not np.any(valid):
                continue
            inside = np.zeros(valid.shape, dtype=bool)
            inside[valid] = mask[v[valid], u[valid]]
            votes[inside] += 1

            if image_path.is_file() and np.any(inside):
                with Image.open(image_path) as raw:
                    frame = np.asarray(
                        raw.convert("RGB").resize(
                            (intr.width, intr.height), Image.Resampling.BILINEAR
                        )
                    )
                idxs = np.flatnonzero(inside)
                color_accum[idxs] += frame[v[idxs], u[idxs]]
                color_count[idxs] += 1

        selected = votes >= max(1, min_votes)
        if int(selected.sum()) < 16:
            # Fall back to any single-view support for sparse scenes.
            selected = votes >= 1
        if int(selected.sum()) < 8:
            raise RuntimeError(
                f"Too few masked SfM points ({int(selected.sum())}); "
                "improve masks or capture denser views"
            )

        pts = xyz[selected]
        if np.any(color_count[selected] > 0):
            cols = np.zeros((int(selected.sum()), 3), dtype=np.uint8)
            for i, idx in enumerate(np.flatnonzero(selected)):
                if color_count[idx] > 0:
                    cols[i] = np.clip(color_accum[idx] / color_count[idx], 0, 255).astype(
                        np.uint8
                    )
                else:
                    cols[i] = rgb[idx] if rgb.shape[0] == xyz.shape[0] else np.array(
                        [180, 180, 180], dtype=np.uint8
                    )
        elif rgb.shape[0] == xyz.shape[0]:
            cols = rgb[selected]
        else:
            cols = np.full((pts.shape[0], 3), 180, dtype=np.uint8)

        # Keep the dense core of votes; long tails inflate room-scale occluders.
        if pts.shape[0] >= 32:
            lo, hi = np.percentile(pts, [25, 75], axis=0)
            keep = np.all((pts >= lo) & (pts <= hi), axis=1)
            if int(keep.sum()) >= 16:
                pts = pts[keep]
                cols = cols[keep]
            else:
                lo, hi = np.percentile(pts, [10, 90], axis=0)
                keep = np.all((pts >= lo) & (pts <= hi), axis=1)
                if int(keep.sum()) >= 8:
                    pts = pts[keep]
                    cols = cols[keep]
        return pts, cols, best_view


def _write_textured_obj(
    mesh: trimesh.Trimesh,
    output_mesh: Path,
    *,
    texture: Image.Image,
    texture_resolution: int,
) -> None:
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    stem = output_mesh.stem
    texture_name = f"{stem}_albedo.png"
    mtl_name = f"{stem}.mtl"
    texture_path = output_mesh.parent / texture_name
    mtl_path = output_mesh.parent / mtl_name

    tex = texture.convert("RGB").resize(
        (texture_resolution, texture_resolution), Image.Resampling.LANCZOS
    )
    tex.save(texture_path)

    # Planar XY UV unwrap in object local bounds.
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mins = verts.min(axis=0)
    extents = np.maximum(verts.max(axis=0) - mins, 1e-6)
    uvs = np.stack(
        [
            (verts[:, 0] - mins[0]) / extents[0],
            (verts[:, 1] - mins[1]) / extents[1],
        ],
        axis=1,
    )

    mtl_path.write_text(
        "\n".join(
            [
                "newmtl scan2usd_albedo",
                "Ka 1.000 1.000 1.000",
                "Kd 1.000 1.000 1.000",
                "Ks 0.000 0.000 0.000",
                "d 1.0",
                "illum 1",
                f"map_Kd {texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lines = [
        f"mtllib {mtl_name}",
        "usemtl scan2usd_albedo",
    ]
    for v in verts:
        lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}")
    for uv in uvs:
        lines.append(f"vt {uv[0]:.8f} {uv[1]:.8f}")
    for face in faces:
        a, b, c = (int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1)
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    output_mesh.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct(args: argparse.Namespace) -> None:
    points, colors, best_view = collect_masked_points(
        images_dir=args.images,
        masks_dir=args.masks,
        colmap_dir=args.colmap,
        min_votes=4,
    )
    # Convex hull is a preview stand-in for dense masked MVS.
    cloud = trimesh.points.PointCloud(points, colors=colors)
    try:
        mesh = cloud.convex_hull
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to build convex hull for {args.instance_id}") from exc
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.faces.shape[0] < 4:
        raise RuntimeError(f"Degenerate object mesh for {args.instance_id}")
    # Preview safety: clamp absurd hulls to a compact box around the core.
    max_extent = float(np.max(mesh.extents))
    if max_extent > 0.6:
        center = mesh.bounding_box.centroid
        extents = np.clip(np.asarray(mesh.extents, dtype=np.float64) * 0.45, 0.08, 0.45)
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)

    if best_view is not None and best_view.is_file():
        texture = Image.open(best_view)
    else:
        mean = tuple(int(x) for x in np.mean(colors, axis=0))
        texture = Image.new("RGB", (64, 64), color=mean)

    _write_textured_obj(
        mesh,
        args.output_mesh,
        texture=texture,
        texture_resolution=max(64, int(args.texture_resolution)),
    )
    report = {
        "instance_id": args.instance_id,
        "num_points": int(points.shape[0]),
        "num_faces": int(mesh.faces.shape[0]),
        "best_view": str(best_view) if best_view else None,
        "output_mesh": str(args.output_mesh.resolve()),
        "method": "masked_colmap_sparse_convex_hull",
    }
    args.output_mesh.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--detail-capture", type=Path, default=None)
    args = parser.parse_args()
    reconstruct(args)


if __name__ == "__main__":
    main()
