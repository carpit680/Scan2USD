"""Estimate a floor plane from COLMAP points and build COLMAP→USD Z-up alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scan2usd.geometry.frames import (
    FRAME_COLMAP,
    FRAME_USD,
    compose_similarity,
    validate_similarity,
)
from scan2usd.reconstruction.colmap_io import (
    export_colmap_to_txt,
    parse_images_txt,
    parse_points3d_txt,
    quat_to_rotmat,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return v / n


def estimate_up_from_colmap_images(images: dict) -> np.ndarray:
    """
    Mean world “sky” direction from COLMAP/OpenCV cameras.

    Camera +Y points down in OpenCV, so camera up is ``R_c2w @ (0, -1, 0)``.
    """
    ups: list[np.ndarray] = []
    for pose in images.values():
        r_w2c = quat_to_rotmat(pose.qvec)
        r_c2w = r_w2c.T
        ups.append(r_c2w @ np.array([0.0, -1.0, 0.0], dtype=np.float64))
    if not ups:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _normalize(np.mean(np.stack(ups, axis=0), axis=0))


@dataclass(frozen=True)
class FloorPlane:
    normal: np.ndarray  # unit, points toward sky
    offset: float  # plane: normal · x = offset
    inlier_count: int
    inlier_ratio: float
    up_hint: np.ndarray


@dataclass(frozen=True)
class FloorAlignment:
    colmap_to_usd: np.ndarray
    floor: FloorPlane
    point_count: int
    report: dict


def fit_floor_plane(
    points: np.ndarray,
    up_hint: np.ndarray,
    *,
    distance_thresh: float = 0.04,
    min_up_alignment: float = 0.75,
    low_percentile: float = 20.0,
    iterations: int = 800,
    seed: int = 0,
) -> FloorPlane:
    """
    RANSAC a dominant near-horizontal plane near the bottom of the cloud.

    Candidate points are restricted to the lower ``low_percentile`` along the
    camera-up axis so countertops/islands do not outvote the true floor.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 32:
        raise RuntimeError("Need at least 32 COLMAP points to estimate a floor plane")
    up = _normalize(np.asarray(up_hint, dtype=np.float64))
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    heights = pts @ up
    # Floor candidates: lower along up (toward gravity). Keep this tight so
    # large furniture planes above the floor cannot dominate RANSAC.
    low_cut = np.percentile(heights, float(low_percentile))
    candidates = pts[heights <= low_cut]
    if candidates.shape[0] < 16:
        candidates = pts

    rng = np.random.default_rng(seed)
    best: tuple[int, np.ndarray, float] | None = None
    for _ in range(iterations):
        sample = candidates[rng.choice(candidates.shape[0], size=3, replace=False)]
        normal = _normalize(np.cross(sample[1] - sample[0], sample[2] - sample[0]))
        if float(np.linalg.norm(normal)) < 1e-8:
            continue
        if abs(float(normal @ up)) < min_up_alignment:
            continue
        if float(normal @ up) < 0.0:
            normal = -normal
        offset = float(np.median(sample @ normal))
        dist = np.abs(pts @ normal - offset)
        score = int(np.count_nonzero(dist < distance_thresh))
        if best is None or score > best[0] or (
            score == best[0] and offset < best[2]
        ):
            best = (score, normal, offset)

    if best is None:
        raise RuntimeError(
            "Failed to find a horizontal floor plane; check COLMAP points / camera up"
        )

    score, normal, offset = best
    # Refine with inliers.
    inliers = pts[np.abs(pts @ normal - offset) < distance_thresh]
    if inliers.shape[0] >= 16:
        centered = inliers - inliers.mean(axis=0)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        refined = _normalize(vt[-1])
        if float(refined @ up) < 0.0:
            refined = -refined
        if abs(float(refined @ up)) >= min_up_alignment:
            normal = refined
            offset = float(np.median(inliers @ normal))
            score = int(np.count_nonzero(np.abs(pts @ normal - offset) < distance_thresh))

    return FloorPlane(
        normal=normal,
        offset=offset,
        inlier_count=score,
        inlier_ratio=float(score) / float(pts.shape[0]),
        up_hint=up,
    )


def rotation_aligning_normal_to_plus_z(normal: np.ndarray) -> np.ndarray:
    """Return R such that ``R @ normal ≈ (0, 0, 1)`` (right-handed)."""
    z_axis = _normalize(np.asarray(normal, dtype=np.float64))
    if float(np.linalg.norm(z_axis)) < 1e-8:
        raise ValueError("normal must be non-zero")
    if abs(float(z_axis[2]) - 1.0) < 1e-8:
        return np.eye(3, dtype=np.float64)
    if abs(float(z_axis[2]) + 1.0) < 1e-8:
        # 180° about X
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)

    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _normalize(np.cross(helper, z_axis))
    if float(np.linalg.norm(x_axis)) < 1e-8:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = _normalize(np.cross(helper, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    # Rows are world axes expressed in USD: R @ world = usd
    return np.stack([x_axis, y_axis, z_axis], axis=0)


def colmap_to_usd_from_floor(
    floor: FloorPlane,
    points: np.ndarray,
    *,
    center_xy: bool = True,
) -> np.ndarray:
    """
    Build a rigid COLMAP→USD similarity: floor normal → +Z, floor plane → Z=0.

    Scale is left at 1.0 (preview / unscaled COLMAP units). Metric scale is approved
    separately when a length anchor is available.
    """
    rotation = rotation_aligning_normal_to_plus_z(floor.normal)
    rotated = (rotation @ np.asarray(points, dtype=np.float64).T).T
    floor_z = float(floor.offset)  # because usd_z = normal · x for this R
    # After R, plane normal·x = offset becomes e_z · (R x) = offset ⇒ z' = offset.
    translation = np.array([0.0, 0.0, -floor_z], dtype=np.float64)
    if center_xy:
        translation[0] -= float(np.median(rotated[:, 0]))
        translation[1] -= float(np.median(rotated[:, 1]))
    return validate_similarity(compose_similarity(rotation=rotation, translation=translation))


def load_colmap_points_and_up(sparse_model: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load XYZ + camera-up hint from a COLMAP sparse model (BIN or TXT)."""
    sparse_model = sparse_model.expanduser().resolve()
    if (sparse_model / "points3D.bin").is_file() or (sparse_model / "cameras.bin").is_file():
        import tempfile

        with tempfile.TemporaryDirectory(prefix="scan2usd_floor_") as tmp:
            txt = Path(tmp)
            export_colmap_to_txt(sparse_model, txt)
            _ids, xyz = parse_points3d_txt(txt / "points3D.txt")
            images = parse_images_txt(txt / "images.txt")
    else:
        _ids, xyz = parse_points3d_txt(sparse_model / "points3D.txt")
        images = parse_images_txt(sparse_model / "images.txt")
    if xyz.shape[0] == 0:
        raise RuntimeError(f"No 3D points under {sparse_model}")
    up = estimate_up_from_colmap_images(images)
    return xyz, up, images


def estimate_floor_alignment(
    sparse_model: Path,
    *,
    distance_thresh: float = 0.04,
    seed: int = 0,
) -> FloorAlignment:
    points, up, _images = load_colmap_points_and_up(sparse_model)
    floor = fit_floor_plane(
        points,
        up,
        distance_thresh=distance_thresh,
        seed=seed,
    )
    matrix = colmap_to_usd_from_floor(floor, points)
    aligned = (matrix[:3, :3] @ points.T).T + matrix[:3, 3]
    report = {
        "source_frame": FRAME_COLMAP,
        "target_frame": FRAME_USD,
        "floor_normal_colmap": floor.normal.tolist(),
        "floor_offset": floor.offset,
        "floor_inliers": floor.inlier_count,
        "floor_inlier_ratio": floor.inlier_ratio,
        "up_hint_colmap": floor.up_hint.tolist(),
        "aligned_z_percentiles": np.percentile(aligned[:, 2], [1, 5, 50, 95, 99]).tolist(),
        "aligned_xy_median": np.median(aligned[:, :2], axis=0).tolist(),
        "distance_thresh_m": distance_thresh,
        "note": "Rigid alignment only (scale=1). Approve metric scale separately if needed.",
    }
    return FloorAlignment(
        colmap_to_usd=matrix,
        floor=floor,
        point_count=int(points.shape[0]),
        report=report,
    )


def write_alignment_json(alignment: FloorAlignment, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "colmap_to_usd": alignment.colmap_to_usd.tolist(),
        "registration_confidence": float(
            np.clip(alignment.floor.inlier_ratio * 1.5, 0.0, 1.0)
        ),
        "method": "floor_plane_ransac",
        "report": alignment.report,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
