"""Draw lifted 3D OBBs in a viser scene (same world frame as Splatfacto)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import viser


# Distinct colors for class ids (RGB 0–255).
_CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 99, 71),
    (50, 205, 50),
    (30, 144, 255),
    (255, 215, 0),
    (186, 85, 211),
    (0, 206, 209),
    (255, 140, 0),
    (220, 20, 60),
    (154, 205, 50),
    (147, 112, 219),
)


def load_objects_3d(
    path: Path,
    *,
    ns_data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ``objects_3d.npz``.

    Returns ``class_id``, ``obb_center``, ``obb_rotation`` (N,3,3), ``obb_half``,
    ``bbox_min``, ``bbox_max`` in transforms.json / sparse_pc frame.
    """
    from scan2usd.synthetic.transforms_io import (
        COORD_FRAME_NERFSTUDIO,
        load_applied_transform,
        transform_aabb_colmap_to_nerfstudio,
    )

    data = np.load(path)
    class_ids = data["class_id"]
    bbox_mins = np.asarray(data["bbox_min"], dtype=np.float64)
    bbox_maxs = np.asarray(data["bbox_max"], dtype=np.float64)
    frame = str(data["coord_frame"]) if "coord_frame" in data else "colmap"

    if frame != COORD_FRAME_NERFSTUDIO and ns_data_dir is not None:
        t = load_applied_transform(ns_data_dir)
        if t is not None:
            mins, maxs, centers, rots, halves = [], [], [], [], []
            for lo, hi in zip(bbox_mins, bbox_maxs):
                bb = transform_aabb_colmap_to_nerfstudio(np.stack([lo, hi], axis=0), t)
                mins.append(bb[0])
                maxs.append(bb[1])
                c = (bb[0] + bb[1]) / 2
                h = (bb[1] - bb[0]) / 2
                centers.append(c)
                rots.append(np.eye(3))
                halves.append(h)
            bbox_mins = np.stack(mins, axis=0)
            bbox_maxs = np.stack(maxs, axis=0)
            centers = np.stack(centers, axis=0)
            rotations = np.stack(rots, axis=0)
            halves = np.stack(halves, axis=0)
            return class_ids, centers, rotations, halves, bbox_mins, bbox_maxs

    if "obb_center" in data:
        centers = np.asarray(data["obb_center"], dtype=np.float64)
        halves = np.asarray(data["obb_half"], dtype=np.float64)
        rotations = np.asarray(data["obb_rotation"], dtype=np.float64)
    else:
        centers = (bbox_mins + bbox_maxs) / 2.0
        halves = (bbox_maxs - bbox_mins) / 2.0
        rotations = np.tile(np.eye(3), (len(class_ids), 1, 1))

    return class_ids, centers, rotations, halves, bbox_mins, bbox_maxs


def apply_dataparser_transform_obbs(
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    dataparser_transform,
    dataparser_scale: float,
    *,
    ns_data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map OBBs from ``transforms.json`` space into splat viewer coordinates."""
    from scan2usd.labeling.obb import transform_obb_rigid
    from scan2usd.synthetic.transforms_io import orient_transform_for_saved_coords

    t_orient = orient_transform_for_saved_coords(dataparser_transform, ns_data_dir)
    scale = float(dataparser_scale)
    out_c, out_r, out_h = [], [], []
    for c, r, h in zip(centers, rotations, half_extents):
        ct, rt, ht = transform_obb_rigid(c, r, h, t_orient, scale)
        out_c.append(ct)
        out_r.append(rt)
        out_h.append(ht)
    return np.stack(out_c), np.stack(out_r), np.stack(out_h)


def class_color(class_id: int) -> tuple[int, int, int]:
    return _CLASS_COLORS[int(class_id) % len(_CLASS_COLORS)]


def overlay_obb_wireframes(
    server: viser.ViserServer,
    class_ids: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    half_extents: np.ndarray,
    *,
    class_names: list[str] | None = None,
    prefix: str = "/scan2usd/boxes",
    viser_scale_ratio: float = 10.0,
) -> list:
    """
    Add oriented wireframe boxes. Returns viser handles (for toggling visibility).
    """
    import trimesh.creation

    from scan2usd.labeling.obb import rotation_to_wxyz

    handles: list = []
    scale = float(viser_scale_ratio)
    for i in range(len(class_ids)):
        half = np.asarray(half_extents[i], dtype=np.float64) * scale
        if np.any(half <= 1e-6):
            continue
        cid = int(class_ids[i])
        color = class_color(cid)
        mesh = trimesh.creation.box(extents=2.0 * half)
        label = ""
        if class_names and 0 <= cid < len(class_names):
            label = class_names[cid]
        name = f"{prefix}/{i}" if not label else f"{prefix}/{i}_{label}"
        wxyz = rotation_to_wxyz(np.asarray(rotations[i], dtype=np.float64))
        handle = server.scene.add_mesh_simple(
            name=name,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.uint32),
            color=color,
            wireframe=True,
            wxyz=wxyz,
            position=tuple(float(x) for x in np.asarray(centers[i]) * scale),
        )
        handles.append(handle)

    if handles:
        show = server.gui.add_checkbox("Show 3D boxes", initial_value=True)

        @show.on_update
        def _toggle(_event) -> None:
            for h in handles:
                h.visible = bool(show.value)

    return handles
