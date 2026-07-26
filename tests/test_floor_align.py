import numpy as np

from scan2usd.geometry.floor_align import (
    colmap_to_usd_from_floor,
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
