from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

COORD_FRAME_COLMAP = "colmap"
COORD_FRAME_NERFSTUDIO = "nerfstudio"


def load_transforms_json(path: Path) -> tuple[list[str], list[np.ndarray], dict[str, Any]]:
    """
    Load Nerfstudio ``transforms.json`` (or ``transforms_train.json``).
    Returns (file_paths relative, c2w 4x4 row-major list, raw meta with fl_x etc.).
    """
    data = json.loads(path.read_text())
    frames = data.get("frames") or []
    paths: list[str] = []
    mats: list[np.ndarray] = []
    for fr in frames:
        fp = fr.get("file_path", "")
        tm = np.array(fr["transform_matrix"], dtype=np.float64)
        if tm.shape == (3, 4):
            m = np.eye(4)
            m[:3, :4] = tm
            tm = m
        paths.append(fp)
        mats.append(tm)
    return paths, mats, data


def c2w_to_colmap_rt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert camera-to-world (nerfstudio row-vector convention) to COLMAP world-to-camera R,t.
    For column vectors: X_w = c2w[:3,:3] X_c + c2w[:3,3]
    COLMAP: X_c = R X_w + t
    => R = R_cw^T, t = - R_cw^T t_wc  with R_cw = c2w[:3,:3], t_wc = c2w[:3,3]
    """
    r_cw = c2w[:3, :3]
    t_wc = c2w[:3, 3]
    r = r_cw.T
    t = -r @ t_wc
    return r, t


def rotmat_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion w,x,y,z (COLMAP)."""
    trace = np.trace(r)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def find_transforms_json(ns_data_dir: Path) -> Path | None:
    for name in ("transforms.json", "transforms_train.json"):
        p = ns_data_dir / name
        if p.is_file():
            return p
    return None


def applied_transform_to_4x4(meta: dict[str, Any]) -> np.ndarray | None:
    """Nerfstudio ``applied_transform`` (COLMAP → saved / viewer coordinates) as 4×4."""
    raw = meta.get("applied_transform")
    if raw is None:
        return None
    t = np.array(raw, dtype=np.float64)
    if t.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = t
        return out
    if t.shape == (4, 4):
        return t
    raise ValueError(f"applied_transform has unexpected shape {t.shape}")


def load_applied_transform(ns_data_dir: Path) -> np.ndarray | None:
    """Load ``applied_transform`` from ``transforms.json`` under a Nerfstudio data dir."""
    tjson = find_transforms_json(ns_data_dir)
    if tjson is None:
        return None
    meta = json.loads(tjson.read_text())
    return applied_transform_to_4x4(meta)


def transform_points_colmap_to_nerfstudio(xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply Nerfstudio COLMAP→viewer transform to N×3 points."""
    pts = np.asarray(xyz, dtype=np.float64)
    r = transform_4x4[:3, :3]
    t = transform_4x4[:3, 3]
    return (pts @ r.T) + t


def aabb_corners(bbox: np.ndarray) -> np.ndarray:
    """Eight corners of axis-aligned bbox with shape (2, 3) min/max."""
    lo, hi = bbox[0], bbox[1]
    return np.array(
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


def transform_aabb_colmap_to_nerfstudio(bbox: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Transform axis-aligned bbox (2,3) min/max from COLMAP world to Nerfstudio world."""
    corners_t = transform_points_colmap_to_nerfstudio(aabb_corners(bbox), transform_4x4)
    return np.stack([corners_t.min(axis=0), corners_t.max(axis=0)], axis=0)


def dataparser_transform_to_4x4(dataparser_transform) -> np.ndarray:
    """Nerfstudio ``dataparser_transform`` (3×4 or 4×4) as a 4×4 matrix."""
    t = np.asarray(dataparser_transform, dtype=np.float64)
    if t.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = t
        return out
    if t.shape == (4, 4):
        return t
    raise ValueError(f"dataparser_transform has unexpected shape {t.shape}")


def transform_points_to_dataparser(
    xyz: np.ndarray,
    dataparser_transform,
    dataparser_scale: float,
) -> np.ndarray:
    """Match NerfstudioDataparser point loading: homogeneous @ T.T, then × scale."""
    pts = np.asarray(xyz, dtype=np.float64)
    t4 = dataparser_transform_to_4x4(dataparser_transform)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (homog @ t4.T)[:, :3] * float(dataparser_scale)


def transform_aabb_to_dataparser(
    bbox: np.ndarray,
    dataparser_transform,
    dataparser_scale: float,
) -> np.ndarray:
    """Transform AABB into the same coordinates as the trained splat / ``ns-viewer`` scene."""
    corners_t = transform_points_to_dataparser(aabb_corners(bbox), dataparser_transform, dataparser_scale)
    return np.stack([corners_t.min(axis=0), corners_t.max(axis=0)], axis=0)


def orient_transform_for_saved_coords(
    dataparser_transform,
    ns_data_dir: Path,
) -> np.ndarray:
    """
    Map ``transforms.json`` / ``sparse_pc.ply`` coordinates → splat viewer coordinates.

    ``dataparser_transform`` is ``T_orient @ T_applied`` (COLMAP → viewer). Points and boxes
    from ``sparse_pc.ply`` are already in saved space (after ``T_applied`` only), so we must
    not apply ``T_applied`` again.
    """
    t_full = dataparser_transform_to_4x4(dataparser_transform)
    t_applied = load_applied_transform(ns_data_dir)
    if t_applied is None:
        return t_full
    return t_full @ np.linalg.inv(t_applied)
