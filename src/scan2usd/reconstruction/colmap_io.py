from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraIntrinsics:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # fx, fy, cx, cy for PINHOLE


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray  # w,x,y,z
    tvec: np.ndarray  # 3
    camera_id: int
    name: str


def quat_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP world-to-camera rotation from quaternion (w,x,y,z)."""
    w, x, y, z = qvec
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def world_to_camera_matrix(qvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t such that X_c = R @ X_w + t (COLMAP convention)."""
    r = quat_to_rotmat(qvec)
    return r, tvec


def project_points(
    xyz_w: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
    intr: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project Nx3 world points to pixel coords.
    Returns (uv, depth) with uv shape (N,2), depth (N,).
    """
    r, t = world_to_camera_matrix(qvec, tvec)
    xc = (r @ xyz_w.T).T + t
    z = xc[:, 2]
    model = intr.model.upper()
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = intr.params[0], intr.params[1], intr.params[2]
        fx = fy = f
    elif model in ("PINHOLE", "OPENCV"):
        # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2 — use pinhole part for lifting
        fx, fy, cx, cy = intr.params[:4]
    else:
        raise ValueError(f"Unsupported camera model: {intr.model}")
    u = fx * (xc[:, 0] / (z + 1e-12)) + cx
    v = fy * (xc[:, 1] / (z + 1e-12)) + cy
    uv = np.stack([u, v], axis=1)
    return uv, z


def parse_cameras_txt(path: Path) -> dict[int, CameraIntrinsics]:
    cams: dict[int, CameraIntrinsics] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        cid = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
        cams[cid] = CameraIntrinsics(cid, model, w, h, params)
    return cams


def parse_images_txt(path: Path) -> dict[str, ImagePose]:
    """Parse COLMAP images.txt (two lines per image)."""
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#") and ln.strip()]
    images: dict[str, ImagePose] = {}
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if len(header) < 9:
            i += 1
            continue
        image_id = int(header[0])
        qw, qx, qy, qz = map(float, header[1:5])
        tx, ty, tz = map(float, header[5:8])
        camera_id = int(header[8])
        name = header[9]
        images[name] = ImagePose(
            image_id=image_id,
            qvec=np.array([qw, qx, qy, qz], dtype=np.float64),
            tvec=np.array([tx, ty, tz], dtype=np.float64),
            camera_id=camera_id,
            name=name,
        )
        i += 2  # skip points2D line
    return images


def parse_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids N,), (N,3) XYZ."""
    ids: list[int] = []
    xyz: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        ids.append(int(parts[0]))
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ids, dtype=np.int64), np.array(xyz, dtype=np.float64)


def export_colmap_to_txt(
    sparse_dir: Path,
    out_dir: Path,
    colmap_bin: str = "colmap",
) -> None:
    """Run COLMAP model_converter to TXT for parsing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin,
        "model_converter",
        "--input_path",
        str(sparse_dir),
        "--output_path",
        str(out_dir),
        "--output_type",
        "TXT",
    ]
    subprocess.run(cmd, check=True)


def read_colmap_model(txt_dir: Path) -> tuple[dict[int, CameraIntrinsics], dict[str, ImagePose], np.ndarray, np.ndarray]:
    cams = parse_cameras_txt(txt_dir / "cameras.txt")
    images = parse_images_txt(txt_dir / "images.txt")
    pids, xyz = parse_points3d_txt(txt_dir / "points3D.txt")
    return cams, images, pids, xyz


def bbox3d_from_points_in_frustum(
    uv: np.ndarray,
    depth: np.ndarray,
    xyz_w: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    min_points: int = 8,
    min_depth: float = 0.05,
) -> np.ndarray | None:
    """Axis-aligned 3D bbox from world points projecting inside 2D rect."""
    mask = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (depth > min_depth)
        & (uv[:, 0] >= x0)
        & (uv[:, 0] <= x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] <= y1)
    )
    pts = xyz_w[mask]
    if pts.shape[0] < min_points:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return np.stack([lo, hi], axis=0)
