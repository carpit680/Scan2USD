from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scan2usd.dataset.split import resolve_label_path
from scan2usd.labeling.detect import read_yolo_label_file, yolo_norm_to_xyxy
from scan2usd.labeling.obb import estimate_scene_up_from_cameras, fit_obb_from_points
from scan2usd.reconstruction.colmap_io import (
    CameraIntrinsics,
    points_in_frustum,
    project_points,
    project_points_c2w,
    read_colmap_model,
)
from scan2usd.synthetic.transforms_io import (
    find_transforms_json,
    load_applied_transform,
    load_transforms_json,
    transform_points_colmap_to_nerfstudio,
)


@dataclass
class Object3D:
    class_id: int
    bbox: np.ndarray  # (2,3) world AABB for YOLO projection
    center: np.ndarray  # (3,) OBB center
    rotation: np.ndarray  # (3,3) box axes → world
    half_extents: np.ndarray  # (3,)
    points: np.ndarray  # supporting SfM points (for merge)
    view_origin: np.ndarray  # (3,) camera center for this observation


def _object_from_frustum_points(
    class_id: int,
    pts: np.ndarray,
    view_origin: np.ndarray,
    world_up: np.ndarray,
    *,
    image_wider_than_tall: bool | None = None,
    percentile: tuple[float, float],
    max_extent_m: float | None,
) -> Object3D:
    center, rotation, half, bbox = fit_obb_from_points(
        pts,
        view_origin=view_origin,
        world_up=world_up,
        image_wider_than_tall=image_wider_than_tall,
        percentile=percentile,
        max_extent_m=max_extent_m,
    )
    return Object3D(
        class_id=class_id,
        bbox=bbox,
        center=center,
        rotation=rotation,
        half_extents=half,
        points=pts.copy(),
        view_origin=np.asarray(view_origin, dtype=np.float64),
    )


def load_ns_sparse_points(ns_data_dir: Path) -> np.ndarray:
    """
    Sparse SfM points in the same frame as ``transforms.json`` (``sparse_pc.ply``).

    Falls back to COLMAP TXT + ``applied_transform`` when the PLY is missing.
    """
    ply = ns_data_dir / "sparse_pc.ply"
    if ply.is_file():
        try:
            import open3d as o3d  # noqa: PLC0415

            pcd = o3d.io.read_point_cloud(str(ply))
            pts = np.asarray(pcd.points, dtype=np.float64)
            if pts.size:
                return pts
        except ImportError:
            pass
    colmap_sparse = ns_data_dir / "colmap" / "sparse" / "0"
    if colmap_sparse.is_dir():
        from scan2usd.reconstruction.colmap_io import export_colmap_to_txt

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            export_colmap_to_txt(colmap_sparse, tmp_p)
            _c, _i, _p, xyz = read_colmap_model(tmp_p)
            t = load_applied_transform(ns_data_dir)
            if t is not None:
                return transform_points_colmap_to_nerfstudio(xyz, t)
            return xyz
    raise FileNotFoundError(
        f"No sparse_pc.ply or colmap/sparse/0 under {ns_data_dir}. Run ``scan2usd reconstruct`` first."
    )


def lift_frame_boxes_to_3d(
    image_name: str,
    labels_dir: Path,
    intr: CameraIntrinsics,
    pose,
    xyz_world: np.ndarray,
    *,
    world_up: np.ndarray | None = None,
    min_points: int = 8,
    depth_trim_mad: float = 3.0,
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
) -> list[Object3D]:
    """Lift using raw COLMAP ``images.txt`` poses (legacy / tests)."""
    rows = read_yolo_label_file(resolve_label_path(labels_dir, image_name))
    if not rows:
        return []
    w, h = intr.width, intr.height
    uv_all, z_all = project_points(xyz_world, pose.qvec, pose.tvec, intr)
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    from scan2usd.reconstruction.colmap_io import world_to_camera_matrix

    r_wc, t_wc = world_to_camera_matrix(pose.qvec, pose.tvec)
    cam_center = -r_wc.T @ t_wc
    out: list[Object3D] = []
    for cid, xc, yc, bw, bh in rows:
        x0, y0, x1, y1 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)
        pts = points_in_frustum(
            uv_all, z_all, xyz_world, x0, y0, x1, y1,
            min_points=min_points, depth_trim_mad=depth_trim_mad,
        )
        if pts is None:
            continue
        out.append(
            _object_from_frustum_points(
                cid,
                pts,
                cam_center,
                world_up,
                image_wider_than_tall=bw >= bh,
                percentile=percentile,
                max_extent_m=max_extent_m,
            )
        )
    return out


def lift_frame_boxes_to_3d_c2w(
    image_name: str,
    labels_dir: Path,
    c2w: np.ndarray,
    intr: CameraIntrinsics,
    xyz_world: np.ndarray,
    *,
    min_points: int = 8,
    depth_trim_mad: float = 3.0,
    inner_margin_frac: float = 0.0,
    depth_percentile: tuple[float, float] | None = (10.0, 90.0),
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
    world_up: np.ndarray | None = None,
) -> list[Object3D]:
    """Lift 2D boxes using Nerfstudio ``transform_matrix`` poses and sparse SfM points."""
    rows = read_yolo_label_file(resolve_label_path(labels_dir, image_name))
    if not rows:
        return []
    w, h = intr.width, intr.height
    c2w = np.asarray(c2w, dtype=np.float64)
    if world_up is None:
        world_up = estimate_scene_up_from_cameras([c2w])
    cam_center = c2w[:3, 3].copy()
    uv_all, z_all = project_points_c2w(xyz_world, c2w, intr)
    out: list[Object3D] = []
    for cid, xc, yc, bw, bh in rows:
        x0, y0, x1, y1 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)
        pts = points_in_frustum(
            uv_all, z_all, xyz_world, x0, y0, x1, y1,
            min_points=min_points,
            depth_trim_mad=depth_trim_mad,
            inner_margin_frac=inner_margin_frac,
            depth_percentile=depth_percentile,
        )
        if pts is None:
            continue
        out.append(
            _object_from_frustum_points(
                cid,
                pts,
                cam_center,
                world_up,
                image_wider_than_tall=bw >= bh,
                percentile=percentile,
                max_extent_m=max_extent_m,
            )
        )
    return out


def lift_one_yolo_box_c2w(
    class_id: int,
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    c2w: np.ndarray,
    intr: CameraIntrinsics,
    xyz_world: np.ndarray,
    uv_all: np.ndarray,
    z_all: np.ndarray,
    *,
    min_points: int = 8,
    depth_trim_mad: float = 3.0,
    inner_margin_frac: float = 0.0,
    depth_percentile: tuple[float, float] | None = (10.0, 90.0),
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
    world_up: np.ndarray | None = None,
) -> Object3D | None:
    """Lift a single 2D detection (used by debug-lift; one box → one 3D object)."""
    w, h = intr.width, intr.height
    c2w = np.asarray(c2w, dtype=np.float64)
    if world_up is None:
        world_up = estimate_scene_up_from_cameras([c2w])
    cam_center = c2w[:3, 3].copy()
    x0, y0, x1, y1 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)
    pts = points_in_frustum(
        uv_all, z_all, xyz_world, x0, y0, x1, y1,
        min_points=min_points,
        depth_trim_mad=depth_trim_mad,
        inner_margin_frac=inner_margin_frac,
        depth_percentile=depth_percentile,
    )
    if pts is None:
        return None
    return _object_from_frustum_points(
        class_id,
        pts,
        cam_center,
        world_up,
        image_wider_than_tall=bw >= bh,
        percentile=percentile,
        max_extent_m=max_extent_m,
    )


def merge_objects(
    objects: list[Object3D],
    *,
    merge_center_dist_m: float,
    world_up: np.ndarray,
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
) -> list[Object3D]:
    """Greedy merge same-class objects with nearby 3D centers."""
    if not objects:
        return []
    objs = sorted(objects, key=lambda o: o.class_id)
    merged: list[Object3D] = []
    for o in objs:
        placed = False
        for j, m in enumerate(merged):
            if m.class_id != o.class_id:
                continue
            if np.linalg.norm(m.center - o.center) <= merge_center_dist_m:
                pts = np.concatenate([m.points, o.points], axis=0)
                n_m, n_o = len(m.points), len(o.points)
                view_origin = (m.view_origin * n_m + o.view_origin * n_o) / max(n_m + n_o, 1)
                merged[j] = _object_from_frustum_points(
                    m.class_id,
                    pts,
                    view_origin,
                    world_up,
                    percentile=percentile,
                    max_extent_m=max_extent_m,
                )
                placed = True
                break
        if not placed:
            merged.append(o)
    return merged


def lift_scene_from_transforms(
    ns_data_dir: Path,
    labels_dir: Path,
    *,
    min_points: int = 8,
    merge_center_dist_m: float = 0.6,
    depth_trim_mad: float = 3.0,
    inner_margin_frac: float = 0.0,
    depth_percentile: tuple[float, float] | None = (10.0, 90.0),
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
) -> list[Object3D]:
    """
    Lift in the same world frame as ``transforms.json`` / ``sparse_pc.ply`` / the trained splat.

    Uses Nerfstudio camera matrices (not raw COLMAP TXT) so 3D boxes align with synthesis and the viewer.
    """
    tjson = find_transforms_json(ns_data_dir)
    if tjson is None:
        raise FileNotFoundError(f"No transforms.json under {ns_data_dir}")
    paths, mats, meta = load_transforms_json(tjson)
    xyz = load_ns_sparse_points(ns_data_dir)
    world_up = estimate_scene_up_from_cameras(mats)
    all_obs: list[Object3D] = []
    for rel_path, c2w in zip(paths, mats):
        name = Path(rel_path).name
        im_path = ns_data_dir / rel_path.lstrip("./")
        if not im_path.is_file():
            continue
        w, h = Image.open(im_path).size
        intr = intrinsics_from_transforms_meta(meta, w, h)
        obs = lift_frame_boxes_to_3d_c2w(
            name,
            labels_dir,
            c2w,
            intr,
            xyz,
            min_points=min_points,
            depth_trim_mad=depth_trim_mad,
            inner_margin_frac=inner_margin_frac,
            depth_percentile=depth_percentile,
            percentile=percentile,
            max_extent_m=max_extent_m,
            world_up=world_up,
        )
        all_obs.extend(obs)
    return merge_objects(
        all_obs,
        merge_center_dist_m=merge_center_dist_m,
        world_up=world_up,
        percentile=percentile,
        max_extent_m=max_extent_m,
    )


def lift_scene(
    colmap_txt_dir: Path,
    labels_dir: Path,
    *,
    min_points: int = 8,
    merge_center_dist_m: float = 0.6,
    ns_data_dir: Path | None = None,
    depth_trim_mad: float = 3.0,
    inner_margin_frac: float = 0.0,
    depth_percentile: tuple[float, float] | None = (10.0, 90.0),
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
) -> list[Object3D]:
    """Prefer ``transforms.json`` lifting when ``ns_data_dir`` is available."""
    if ns_data_dir is not None and find_transforms_json(ns_data_dir) is not None:
        return lift_scene_from_transforms(
            ns_data_dir,
            labels_dir,
            min_points=min_points,
            merge_center_dist_m=merge_center_dist_m,
            depth_trim_mad=depth_trim_mad,
            inner_margin_frac=inner_margin_frac,
            depth_percentile=depth_percentile,
            percentile=percentile,
            max_extent_m=max_extent_m,
        )
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
            depth_trim_mad=depth_trim_mad,
            percentile=percentile,
            max_extent_m=max_extent_m,
        )
        all_obs.extend(obs)
    world_up_colmap = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    merged = merge_objects(
        all_obs,
        merge_center_dist_m=merge_center_dist_m,
        world_up=world_up_colmap,
        percentile=percentile,
        max_extent_m=max_extent_m,
    )
    if ns_data_dir is not None:
        t = load_applied_transform(ns_data_dir)
        tjson = find_transforms_json(ns_data_dir)
        if t is not None and tjson is not None:
            _paths, mats, _meta = load_transforms_json(tjson)
            up_ns = estimate_scene_up_from_cameras(mats)
            remerged: list[Object3D] = []
            for o in merged:
                pts_t = transform_points_colmap_to_nerfstudio(o.points, t)
                vo_t = transform_points_colmap_to_nerfstudio(o.view_origin[None], t)[0]
                up_t = t[:3, :3] @ up_ns
                up_t = up_t / (np.linalg.norm(up_t) + 1e-12)
                remerged.append(
                    _object_from_frustum_points(
                        o.class_id,
                        pts_t,
                        vo_t,
                        up_t,
                        percentile=percentile,
                        max_extent_m=max_extent_m,
                    )
                )
            merged = remerged
    return merged


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
    """Project using Nerfstudio camera-to-world."""
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
    uv, z = project_points_c2w(corners, c2w, intr)
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
