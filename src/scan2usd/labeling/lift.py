from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan2usd.dataset.split import resolve_label_path
from scan2usd.labeling.detect import read_yolo_label_file, yolo_norm_to_xyxy
from scan2usd.reconstruction.colmap_io import (
    CameraIntrinsics,
    bbox3d_from_points_in_frustum,
    project_points,
    read_colmap_model,
)
from scan2usd.synthetic.transforms_io import c2w_to_colmap_rt, rotmat_to_quat_wxyz


@dataclass
class Object3D:
    class_id: int
    bbox: np.ndarray  # (2,3) min/max corners world AABB


def lift_frame_boxes_to_3d(
    image_name: str,
    labels_path: Path,
    intr: CameraIntrinsics,
    pose,
    xyz_world: np.ndarray,
    *,
    min_points: int = 8,
) -> list[Object3D]:
    """Lift each 2D YOLO box to a 3D AABB using COLMAP points projecting inside the box."""
    rows = read_yolo_label_file(resolve_label_path(labels_path, image_name))
    if not rows:
        return []
    w, h = intr.width, intr.height
    uv_all, z_all = project_points(xyz_world, pose.qvec, pose.tvec, intr)
    out: list[Object3D] = []
    for cid, xc, yc, bw, bh in rows:
        x0, y0, x1, y1 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)
        bb = bbox3d_from_points_in_frustum(
            uv_all,
            z_all,
            xyz_world,
            x0,
            y0,
            x1,
            y1,
            min_points=min_points,
        )
        if bb is None:
            continue
        out.append(Object3D(class_id=cid, bbox=bb))
    return out


def merge_objects(
    objects: list[Object3D],
    *,
    merge_center_dist_m: float,
) -> list[Object3D]:
    """Greedy merge same-class objects with nearby 3D centers."""
    if not objects:
        return []
    objs = sorted(objects, key=lambda o: o.class_id)
    merged: list[Object3D] = []
    for o in objs:
        c = (o.bbox[0] + o.bbox[1]) / 2.0
        placed = False
        for j, m in enumerate(merged):
            if m.class_id != o.class_id:
                continue
            mc = (m.bbox[0] + m.bbox[1]) / 2.0
            if np.linalg.norm(mc - c) <= merge_center_dist_m:
                lo = np.minimum(m.bbox[0], o.bbox[0])
                hi = np.maximum(m.bbox[1], o.bbox[1])
                merged[j] = Object3D(class_id=m.class_id, bbox=np.stack([lo, hi], axis=0))
                placed = True
                break
        if not placed:
            merged.append(o)
    return merged


def lift_scene(
    colmap_txt_dir: Path,
    labels_dir: Path,
    *,
    min_points: int = 8,
    merge_center_dist_m: float = 0.35,
) -> list[Object3D]:
    cams, images, _pids, xyz = read_colmap_model(colmap_txt_dir)
    all_obs: list[Object3D] = []
    for name, pose in images.items():
        intr = cams[pose.camera_id]
        obs = lift_frame_boxes_to_3d(
            name,
            labels_dir,
            intr,
            pose,
            xyz,
            min_points=min_points,
        )
        all_obs.extend(obs)
    return merge_objects(all_obs, merge_center_dist_m=merge_center_dist_m)


def project_aabb_to_yolo_line(
    bbox: np.ndarray,
    class_id: int,
    qvec: np.ndarray,
    tvec: np.ndarray,
    intr: CameraIntrinsics,
) -> tuple[int, float, float, float, float] | None:
    """Project world-axis AABB to YOLO xywh using pinhole model."""
    lo, hi = bbox[0], bbox[1]
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )
    uv, z = project_points(corners, qvec, tvec, intr)
    if np.any(z <= 1e-4):
        return None
    u, v = uv[:, 0], uv[:, 1]
    x0, x1 = float(u.min()), float(u.max())
    y0, y1 = float(v.min()), float(v.max())
    w, h = intr.width, intr.height
    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return None
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    bw = max(1e-6, (x1 - x0) / w)
    bh = max(1e-6, (y1 - y0) / h)
    xc = ((x0 + x1) / 2) / w
    yc = ((y0 + y1) / 2) / h
    return class_id, xc, yc, bw, bh


def project_aabb_to_yolo_line_c2w(
    bbox: np.ndarray,
    class_id: int,
    c2w: np.ndarray,
    intr: CameraIntrinsics,
) -> tuple[int, float, float, float, float] | None:
    """Project using camera-to-world (Nerfstudio) instead of COLMAP q,t."""
    r, t = c2w_to_colmap_rt(c2w)
    qvec = rotmat_to_quat_wxyz(r)
    return project_aabb_to_yolo_line(bbox, class_id, qvec, t, intr)


def intrinsics_from_transforms_meta(meta: dict, width: int, height: int) -> CameraIntrinsics:
    """Build a PINHOLE model from Nerfstudio ``transforms.json`` metadata."""
    if "fl_x" in meta and "fl_y" in meta:
        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta.get("cx", width / 2))
        cy = float(meta.get("cy", height / 2))
    elif "camera_angle_x" in meta:
        ang = float(meta["camera_angle_x"])
        fx = width / (2 * np.tan(ang / 2))
        fy = fx
        cx = width / 2
        cy = height / 2
    else:
        fx = fy = 0.8 * width
        cx = width / 2
        cy = height / 2
    params = np.array([fx, fy, cx, cy], dtype=np.float64)
    return CameraIntrinsics(0, "PINHOLE", width, height, params)
