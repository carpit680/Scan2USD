from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scan2usd.synthetic.transforms_io import rotmat_to_quat_wxyz


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation; quaternions wxyz."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize(q0 + t * (q1 - q0))
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-12)
    s1 = sin_theta / (sin_theta_0 + 1e-12)
    return _normalize(s0 * q0 + s1 * q1)


def interpolate_c2w(c2w_list: list[np.ndarray], alphas: np.ndarray) -> list[np.ndarray]:
    """Piecewise geodesic-ish interpolation on translation + slerp on rotation."""
    if not c2w_list:
        return []
    n = len(c2w_list)
    out: list[np.ndarray] = []
    for a in alphas:
        a = float(np.clip(a, 0.0, 1.0))
        pos = a * (n - 1)
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, n - 1)
        u = pos - i0
        t0 = c2w_list[i0][:3, 3]
        t1 = c2w_list[i1][:3, 3]
        trans = (1 - u) * t0 + u * t1
        r0 = c2w_list[i0][:3, :3]
        r1 = c2w_list[i1][:3, :3]
        q0 = rotmat_to_quat_wxyz(r0)
        q1 = rotmat_to_quat_wxyz(r1)
        q = slerp(q0, q1, u)
        w, x, y, z = q
        # quat wxyz to rotmat
        rot = np.array(
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
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = trans
        out.append(m)
    return out


def jitter_c2w(
    c2w: np.ndarray,
    *,
    pos_sigma: float,
    height_sigma: float,
    max_rot_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    m = c2w.copy()
    jitter = rng.normal(scale=pos_sigma, size=3)
    jitter[1] += rng.normal(scale=height_sigma)
    m[:3, 3] += jitter
    # small euler
    rx = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    ry = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    rz = math.radians(rng.uniform(-max_rot_deg, max_rot_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    r_delta = rzm @ rym @ rxm
    m[:3, :3] = m[:3, :3] @ r_delta
    return m


def sample_novel_poses(
    c2w_trajectory: list[np.ndarray],
    *,
    num_poses: int,
    position_jitter_m: float,
    height_jitter_m: float,
    max_rotation_deg: float,
    interpolation_keyframes: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(c2w_trajectory) < 2:
        return list(c2w_trajectory)
    # base alphas along path + midpoints
    base = np.linspace(0.0, 1.0, max(interpolation_keyframes, 2))
    traj = interpolate_c2w(c2w_trajectory, base)
    alphas = rng.uniform(0.0, 1.0, size=num_poses)
    poses = interpolate_c2w(traj, alphas)
    out = [
        jitter_c2w(
            p,
            pos_sigma=position_jitter_m,
            height_sigma=height_jitter_m,
            max_rot_deg=max_rotation_deg,
            rng=rng,
        )
        for p in poses
    ]
    return out


def fov_deg_from_meta(meta: dict[str, Any], width: int, height: int) -> float:
    if "camera_angle_x" in meta:
        ang = float(meta["camera_angle_x"])
        return float(math.degrees(2 * math.atan(math.tan(ang / 2) * (width / max(height, 1)))))
    if "fl_x" in meta and "fl_y" in meta:
        fl = (float(meta["fl_x"]) + float(meta["fl_y"])) / 2.0
        return float(math.degrees(2 * math.atan(width / (2 * fl))))
    return 60.0


def write_nerfstudio_camera_path(
    c2w_matrices: list[np.ndarray],
    *,
    width: int,
    height: int,
    meta: dict[str, Any],
    out_path: Path,
) -> None:
    """Write JSON for ``ns-render camera-path``."""
    fov = fov_deg_from_meta(meta, width, height)
    seconds = max(1, len(c2w_matrices)) / 24.0
    cam_path = []
    for m in c2w_matrices:
        # row-vector convention: each row is a 4-vector for glm-style; nerfstudio expects list of 4 lists
        c2w_list = m.tolist()
        cam_path.append(
            {
                "camera_to_world": c2w_list,
                "fov": fov,
                "aspect": width / max(height, 1),
            }
        )
    doc = {
        "camera_type": "perspective",
        "render_height": height,
        "render_width": width,
        "fps": 24,
        "seconds": seconds,
        "camera_path": cam_path,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
