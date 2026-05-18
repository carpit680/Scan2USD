from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
