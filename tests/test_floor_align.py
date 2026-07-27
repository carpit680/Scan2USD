from pathlib import Path

import numpy as np
import pytest

from scan2usd.geometry.floor_align import (
    colmap_to_usd_from_floor,
    estimate_floor_alignment,
    fit_floor_plane,
    rotation_aligning_normal_to_plus_z,
)


def test_rotation_aligns_normal_to_plus_z() -> None:
    normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rotation = rotation_aligning_normal_to_plus_z(normal)
    aligned = rotation @ normal
    np.testing.assert_allclose(aligned, [0.0, 0.0, 1.0], atol=1e-8)
    assert np.linalg.det(rotation) > 0.0


def test_floor_alignment_puts_plane_at_z0() -> None:
    rng = np.random.default_rng(0)
    # Floor at y = -1.5 in a Y-up COLMAP-like cloud; clutter above the floor.
    floor_xy = rng.uniform(-2.0, 2.0, size=(400, 2))
    floor = np.column_stack(
        [floor_xy[:, 0], np.full(400, -1.5) + rng.normal(0.0, 0.005, 400), floor_xy[:, 1]]
    )
    clutter = rng.uniform([-1.5, -1.0, -1.5], [1.5, 1.5, 1.5], size=(200, 3))
    points = np.vstack([floor, clutter])
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    floor_plane = fit_floor_plane(points, up_hint, distance_thresh=0.03, seed=1)
    assert floor_plane.inlier_ratio > 0.4
    np.testing.assert_allclose(np.abs(floor_plane.normal @ up_hint), 1.0, atol=0.05)

    matrix = colmap_to_usd_from_floor(floor_plane, points)
    aligned = (matrix[:3, :3] @ floor.T).T + matrix[:3, 3]
    assert float(np.median(np.abs(aligned[:, 2]))) < 0.02
    assert abs(float(np.median(aligned[:, 0]))) < 0.25
    assert abs(float(np.median(aligned[:, 1]))) < 0.25


def _synthetic_scene(floor_ratio: float, *, n: int = 500, seed: int = 3):
    """Y-up cloud with a controllable fraction of true floor points at y=-1."""
    rng = np.random.default_rng(seed)
    n_floor = int(n * floor_ratio)
    floor_xy = rng.uniform(-2.0, 2.0, size=(n_floor, 2))
    floor = np.column_stack(
        [floor_xy[:, 0], np.full(n_floor, -1.0) + rng.normal(0, 0.004, n_floor), floor_xy[:, 1]]
    )
    clutter = rng.uniform([-2.0, -0.8, -2.0], [2.0, 1.5, 2.0], size=(n - n_floor, 3))
    return np.vstack([floor, clutter])


def _patch_loader(monkeypatch, points: np.ndarray, camera_y: float) -> None:
    from types import SimpleNamespace

    from scan2usd.geometry import floor_align

    # Cameras looking forward at height camera_y (identity rotation, Y-up world).
    images = {
        f"im{i}": SimpleNamespace(
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([0.0, -camera_y, 0.0]),
        )
        for i in range(6)
    }
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    monkeypatch.setattr(
        floor_align,
        "load_colmap_points_and_up",
        lambda _path: (points, up, images),
    )


def test_estimate_rejects_plane_floating_through_scene(monkeypatch, tmp_path: Path) -> None:
    """
    The desk-scan failure mode: a plane with a third of the cloud beneath it.

    Points are centred on y=0 with no floor at all, so any fitted plane cuts
    through the middle and much of the scene ends up below it.
    """
    rng = np.random.default_rng(5)
    points = rng.uniform([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], size=(600, 3))
    _patch_loader(monkeypatch, points, camera_y=5.0)
    with pytest.raises(RuntimeError, match="not the ground"):
        estimate_floor_alignment(tmp_path, max_points_below=0.10)


def test_estimate_accepts_sparse_but_real_floor(monkeypatch, tmp_path: Path) -> None:
    """
    A real floor is often only a small share of points (the kitchen scan: 6%).

    It must still be accepted, because the scene rests on top of it.
    """
    points = _synthetic_scene(0.08)
    _patch_loader(monkeypatch, points, camera_y=0.5)
    alignment = estimate_floor_alignment(
        tmp_path, min_inlier_ratio=0.02, max_points_below=0.10
    )
    assert alignment.report["points_below_floor_fraction"] <= 0.10
    assert alignment.report["cameras_above_floor_fraction"] >= 0.9


def test_estimate_accepts_good_floor(monkeypatch, tmp_path: Path) -> None:
    points = _synthetic_scene(0.6)
    _patch_loader(monkeypatch, points, camera_y=0.5)
    alignment = estimate_floor_alignment(tmp_path, min_inlier_ratio=0.02)
    assert alignment.floor.inlier_ratio > 0.3
    assert alignment.report["cameras_above_floor_fraction"] >= 0.9


def test_estimate_rejects_floor_above_cameras(monkeypatch, tmp_path: Path) -> None:
    """Plane above the cameras (ceiling/table misfit) is rejected."""
    points = _synthetic_scene(0.6)
    _patch_loader(monkeypatch, points, camera_y=-3.0)
    with pytest.raises(RuntimeError, match="above"):
        estimate_floor_alignment(tmp_path, min_inlier_ratio=0.0)
