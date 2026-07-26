"""Oriented 3D bounding boxes from point clusters."""

from __future__ import annotations

import numpy as np

from scan2usd.synthetic.transforms_io import aabb_corners


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def estimate_scene_up_from_cameras(c2w_mats: list[np.ndarray]) -> np.ndarray:
    """
    Mean “up” direction in Nerfstudio ``transform_matrix`` world (OpenGL camera +Y → world).
    """
    ups: list[np.ndarray] = []
    for c2w in c2w_mats:
        r = np.asarray(c2w[:3, :3], dtype=np.float64)
        ups.append(r @ np.array([0.0, 1.0, 0.0]))
    return _normalize(np.mean(ups, axis=0))


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(rotation)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _maybe_swap_horizontal_axes(
    rotation: np.ndarray,
    half: np.ndarray,
    *,
    image_wider_than_tall: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Match OBB horizontal extent ordering to the 2D YOLO box shape."""
    if image_wider_than_tall and half[0] < half[1]:
        swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotation = rotation @ swap
        half = np.array([half[1], half[0], half[2]])
    if (not image_wider_than_tall) and half[0] > half[1]:
        swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotation = rotation @ swap
        half = np.array([half[1], half[0], half[2]])
    return _orthonormalize_rotation(rotation), half


def fit_obb_view_aligned(
    pts: np.ndarray,
    view_origin: np.ndarray,
    world_up: np.ndarray,
    *,
    image_wider_than_tall: bool | None = None,
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    OBB with axes tied to gravity (scene up) and the viewing camera.

    Local axes (columns of ``rotation``): right, forward (horizontal toward object), up.
    Works much better than PCA on sparse, noisy SfM points for furniture-like objects.
    """
    pts = np.asarray(pts, dtype=np.float64)
    world_up = _normalize(np.asarray(world_up, dtype=np.float64))
    view_origin = np.asarray(view_origin, dtype=np.float64)
    center = np.median(pts, axis=0)

    to_obj = center - view_origin
    depth = _normalize(to_obj)
    forward = depth - np.dot(depth, world_up) * world_up
    if float(np.linalg.norm(forward)) < 1e-3:
        forward = np.cross(world_up, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(forward)) < 1e-3:
            forward = np.cross(world_up, np.array([0.0, 1.0, 0.0]))
    forward = _normalize(forward)
    right = _normalize(np.cross(forward, world_up))
    forward = _normalize(np.cross(world_up, right))
    rotation = np.column_stack([right, forward, world_up])
    rotation = _orthonormalize_rotation(rotation)

    local = (pts - center) @ rotation
    lo = np.percentile(local, percentile[0], axis=0)
    hi = np.percentile(local, percentile[1], axis=0)
    if max_extent_m is not None:
        extent = hi - lo
        extent = np.minimum(extent, max_extent_m)
        mid = (lo + hi) / 2.0
        lo = mid - extent / 2.0
        hi = mid + extent / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-3)
    if image_wider_than_tall is not None:
        rotation, half = _maybe_swap_horizontal_axes(
            rotation, half, image_wider_than_tall=image_wider_than_tall
        )
    center = center + rotation @ ((lo + hi) / 2.0)
    bbox = np.stack(
        [obb_corners(center, rotation, half).min(axis=0), obb_corners(center, rotation, half).max(axis=0)],
        axis=0,
    )
    return center, rotation, half, bbox


def fit_obb_from_points(
    pts: np.ndarray,
    *,
    view_origin: np.ndarray | None = None,
    world_up: np.ndarray | None = None,
    image_wider_than_tall: bool | None = None,
    percentile: tuple[float, float] = (5.0, 95.0),
    max_extent_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prefer view-aligned OBB when camera pose and scene up are known."""
    if view_origin is not None and world_up is not None:
        return fit_obb_view_aligned(
            pts,
            view_origin,
            world_up,
            image_wider_than_tall=image_wider_than_tall,
            percentile=percentile,
            max_extent_m=max_extent_m,
        )
  # PCA fallback
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 3:
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        center = (lo + hi) / 2.0
        half = np.maximum((hi - lo) / 2.0, 1e-3)
        return center, np.eye(3), half, np.stack([lo, hi], axis=0)

    center = pts.mean(axis=0)
    x = pts - center
    cov = (x.T @ x) / max(pts.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    rotation = _orthonormalize_rotation(evecs[:, order])
    local = x @ rotation
    lo = np.percentile(local, percentile[0], axis=0)
    hi = np.percentile(local, percentile[1], axis=0)
    if max_extent_m is not None:
        extent = hi - lo
        extent = np.minimum(extent, max_extent_m)
        mid = (lo + hi) / 2.0
        lo = mid - extent / 2.0
        hi = mid + extent / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-3)
    center = center + rotation @ ((lo + hi) / 2.0)
    bbox = np.stack(
        [obb_corners(center, rotation, half).min(axis=0), obb_corners(center, rotation, half).max(axis=0)],
        axis=0,
    )
    return center, rotation, half, bbox


def obb_corners(center: np.ndarray, rotation: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Eight world corners of an OBB."""
    lo = -half_extents
    hi = half_extents
    local = aabb_corners(np.stack([lo, hi], axis=0))
    return center + (rotation @ local.T).T


def rotation_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → viser quaternion (w, x, y, z)."""
    import viser.transforms as vtf

    return tuple(float(x) for x in vtf.SO3.from_matrix(_orthonormalize_rotation(rotation)).wxyz)


def transform_obb_rigid(
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    transform_4x4: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigid dataparser transform + uniform scale (preserves OBB orientation)."""
    t4 = np.asarray(transform_4x4, dtype=np.float64)
    if t4.shape == (3, 4):
        r_lin = t4[:3, :3]
        t = t4[:3, 3]
    else:
        r_lin = t4[:3, :3]
        t = t4[:3, 3]
    scale = float(scale)
    center_n = (r_lin @ np.asarray(center, dtype=np.float64) + t) * scale
    rotation_n = _orthonormalize_rotation(r_lin @ np.asarray(rotation, dtype=np.float64))
    half_n = np.asarray(half_extents, dtype=np.float64) * scale
    return center_n, rotation_n, half_n


def transform_obb(
    center: np.ndarray,
    rotation: np.ndarray,
    half_extents: np.ndarray,
    transform_4x4: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply dataparser transform to an OBB (rigid)."""
    return transform_obb_rigid(center, rotation, half_extents, transform_4x4, scale)
