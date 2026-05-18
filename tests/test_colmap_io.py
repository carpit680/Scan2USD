from pathlib import Path

import numpy as np

from scan2usd.reconstruction.colmap_io import (
    CameraIntrinsics,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
    project_points,
)


def test_parse_minimal_colmap_txt(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "# comment\n"
        "1 PINHOLE 640 480 500 500 320 240\n"
    )
    (tmp_path / "images.txt").write_text(
        "#\n"
        "1 1 0 0 0 0 0 0 1 frame_000.jpg\n"
        "0 0 -1\n"
    )
    (tmp_path / "points3D.txt").write_text(
        "#\n"
        "1 1 2 3 255 0 0 0.1 1 1\n"
    )
    cams = parse_cameras_txt(tmp_path / "cameras.txt")
    imgs = parse_images_txt(tmp_path / "images.txt")
    ids, xyz = parse_points3d_txt(tmp_path / "points3D.txt")
    assert 1 in cams
    assert "frame_000.jpg" in imgs
    assert ids.shape == (1,)
    assert xyz.shape == (1, 3)


def test_project_identity_camera() -> None:
    intr = CameraIntrinsics(1, "PINHOLE", 640, 480, np.array([500.0, 500.0, 320.0, 240.0]))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = np.zeros(3)
    pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    uv, z = project_points(pts, q, t, intr)
    assert z[0] > 0
    assert abs(uv[0, 0] - 320.0) < 1e-3
    assert abs(uv[0, 1] - 240.0) < 1e-3


def test_project_opencv_uses_pinhole_intrinsics() -> None:
    params = np.array([500.0, 500.0, 320.0, 240.0, 0.01, -0.02, 0.0, 0.0])
    pinhole = CameraIntrinsics(1, "PINHOLE", 640, 480, params[:4])
    opencv = CameraIntrinsics(2, "OPENCV", 640, 480, params)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = np.zeros(3)
    pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    uv_p, _ = project_points(pts, q, t, pinhole)
    uv_o, _ = project_points(pts, q, t, opencv)
    np.testing.assert_allclose(uv_o, uv_p, rtol=0, atol=1e-6)
