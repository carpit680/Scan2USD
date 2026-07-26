import numpy as np

from scan2usd.labeling.obb import fit_obb_from_points, fit_obb_view_aligned, rotation_to_wxyz


def test_fit_obb_view_aligned_uses_camera():
    pts = np.random.randn(40, 3) * 0.2 + np.array([2.0, 0.0, 0.5])
    cam = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    _c, rotation, half, _bbox = fit_obb_view_aligned(pts, cam, up)
    # forward axis (column 1) should point roughly toward +X
    assert rotation[0, 1] > 0.5
    wxyz = rotation_to_wxyz(rotation)
    assert len(wxyz) == 4


def test_fit_obb_from_diagonal_points():
    t = np.linspace(0, 1, 50)
    pts = np.stack([t, t * 0.5, np.zeros_like(t)], axis=1)
    center, rotation, half, bbox = fit_obb_from_points(pts)
    assert half[0] > half[1] * 0.5
    wxyz = rotation_to_wxyz(rotation)
    assert len(wxyz) == 4
