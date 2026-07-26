"""Debug 2D→3D lift by overlaying labels, SfM inliers, and reprojected boxes on real frames."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.dataset.split import resolve_label_path
from scan2usd.labeling.detect import read_yolo_label_file, yolo_norm_to_xyxy
from scan2usd.labeling.lift import (
    intrinsics_from_transforms_meta,
    lift_one_yolo_box_c2w,
    load_ns_sparse_points,
    project_aabb_to_yolo_line_c2w,
)
from scan2usd.labeling.obb import obb_corners
from scan2usd.reconstruction.colmap_io import points_in_frustum, project_points_c2w
from scan2usd.synthetic.transforms_io import find_transforms_json, load_transforms_json


def _yolo_iou(
    proj: tuple[int, float, float, float, float],
    label: tuple[int, float, float, float, float],
    w: int,
    h: int,
) -> float:
    _, xc, yc, bw, bh = proj
    x0, y0, x1, y1 = yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)
    if label[0] != proj[0]:
        return 0.0
    lx0, ly0, lx1, ly1 = yolo_norm_to_xyxy(label[1], label[2], label[3], label[4], w, h)
    ix0, iy0 = max(x0, lx0), max(y0, ly0)
    ix1, iy1 = min(x1, lx1), min(y1, ly1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (x1 - x0) * (y1 - y0) + (lx1 - lx0) * (ly1 - ly0) - inter + 1e-6
    return float(inter / union)


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thick: int = 2) -> None:
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)


def _draw_aabb_reproj(
    img: np.ndarray,
    obj,
    c2w: np.ndarray,
    intr,
) -> bool:
    """Draw axis-aligned 2D rect from projected 3D AABB (same metric as IoU)."""
    proj = project_aabb_to_yolo_line_c2w(obj.bbox, obj.class_id, c2w, intr)
    if proj is None:
        return False
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in yolo_norm_to_xyxy(proj[1], proj[2], proj[3], proj[4], w, h)]
    _draw_rect(img, x0, y0, x1, y1, (0, 0, 255), 2)
    return True


def _draw_projected_obb(
    img: np.ndarray,
    obj,
    c2w: np.ndarray,
    intr,
) -> bool:
    """Draw projected OBB wireframe (may differ from AABB rect when box is rotated)."""
    corners = obb_corners(obj.center, obj.rotation, obj.half_extents)
    uv, z = project_points_c2w(corners, c2w, intr)
    if np.any(z <= 1e-4):
        return False
    pts = []
    h, w = img.shape[:2]
    for u, v in uv:
        if np.isfinite(u) and np.isfinite(v):
            pts.append([int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))])
    if len(pts) < 3:
        return False
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    cv2.polylines(img, [hull], isClosed=True, color=(0, 128, 255), thickness=1)
    return True


def _banner(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run_lift_debug(
    cfg: SceneConfig,
    out_dir: Path,
    *,
    max_frames: int = 16,
    frame_stride: int = 25,
    min_points: int = 8,
    depth_trim_mad: float = 3.0,
    inner_margin_frac: float = 0.12,
    depth_percentile: tuple[float, float] | None = (10.0, 90.0),
) -> Path:
    """
    Write per-detection side-by-side images and ``summary.json`` under ``out_dir``.

    Layout (left → right):
    - **YOLO** — pseudo-label only (reference; usually better)
    - **Lift** — sparse SfM inliers (cyan) + 3D box reprojected (red AABB, blue OBB hull)
    """
    ns = cfg.nerfstudio_data_dir
    labels_dir = cfg.workspace_dir / "labels_real"
    tjson = find_transforms_json(ns)
    if tjson is None:
        raise FileNotFoundError(f"No transforms.json under {ns}")
    paths, mats, meta = load_transforms_json(tjson)
    xyz = load_ns_sparse_points(ns)

    lift_cfg = cfg.lift or {}
    percentile = tuple(lift_cfg.get("percentile", [5.0, 95.0]))
    max_extent = lift_cfg.get("max_extent_m")
    max_extent_m = None if max_extent is None else float(max_extent)

    out_dir.mkdir(parents=True, exist_ok=True)
    readme = out_dir / "README.txt"
    readme.write_text(
        "Lift debug — side by side (same camera as YOLO label)\n"
        "  LEFT   = YOLO pseudo-label only (trust this for 2D quality)\n"
        "  RIGHT  = 3D lift attempt for THAT detection only\n"
        "           CYAN dots = sparse SfM points used inside the 2D box\n"
        "           RED rect  = 3D AABB reprojected (IoU metric)\n"
        "           BLUE hull = oriented 3D box reprojected\n"
        "\n"
        "If RIGHT does not hug LEFT: lift is wrong (sparse points / depth / merge),\n"
        "  not a splat-viewer issue. YOLO being better is expected when inliers\n"
        "  include walls/floor along the ray or n_sfm_inliers is huge (see summary).\n",
        encoding="utf-8",
    )

    records: list[dict] = []
    written = 0
    for idx in range(0, len(paths), max(1, frame_stride)):
        if written >= max_frames:
            break
        rel = paths[idx]
        name = Path(rel).name
        label_path = resolve_label_path(labels_dir, name)
        if not label_path.is_file():
            continue
        labels = read_yolo_label_file(label_path)
        if not labels:
            continue

        im_path = ns / rel.lstrip("./")
        if not im_path.is_file():
            continue
        bgr = cv2.cvtColor(np.array(Image.open(im_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        intr = intrinsics_from_transforms_meta(meta, w, h)
        c2w = mats[idx]
        uv_all, z_all = project_points_c2w(xyz, c2w, intr)

        for li, (cid, xc, yc, bw, bh) in enumerate(labels):
            x0, y0, x1, y1 = [int(v) for v in yolo_norm_to_xyxy(xc, yc, bw, bh, w, h)]
            yolo_panel = bgr.copy()
            _draw_rect(yolo_panel, x0, y0, x1, y1, (0, 255, 0), 2)
            _banner(yolo_panel, "YOLO (reference)")

            lift_panel = bgr.copy()
            inliers = points_in_frustum(
                uv_all, z_all, xyz, float(x0), float(y0), float(x1), float(y1),
                min_points=1,
                depth_trim_mad=depth_trim_mad,
                inner_margin_frac=inner_margin_frac,
                depth_percentile=depth_percentile,
            )
            n_in = 0 if inliers is None else int(inliers.shape[0])
            if inliers is not None:
                uvi, _ = project_points_c2w(inliers, c2w, intr)
                for u, v in uvi:
                    if np.isfinite(u) and np.isfinite(v):
                        cv2.circle(
                            lift_panel,
                            (int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))),
                            2,
                            (255, 255, 0),
                            -1,
                        )

            obj = lift_one_yolo_box_c2w(
                cid, xc, yc, bw, bh,
                c2w, intr, xyz, uv_all, z_all,
                min_points=min_points,
                depth_trim_mad=depth_trim_mad,
                inner_margin_frac=inner_margin_frac,
                depth_percentile=depth_percentile,
                percentile=percentile,
                max_extent_m=max_extent_m,
            )
            iou = 0.0
            drew = False
            if obj is not None:
                drew = _draw_aabb_reproj(lift_panel, obj, c2w, intr)
                _draw_projected_obb(lift_panel, obj, c2w, intr)
                proj = project_aabb_to_yolo_line_c2w(obj.bbox, cid, c2w, intr)
                if proj is not None:
                    iou = _yolo_iou(proj, (cid, xc, yc, bw, bh), w, h)

            warn = ""
            if n_in > 400:
                warn = "  MANY INLIERS"
            _banner(lift_panel, f"Lift  pts={n_in}  iou={iou:.2f}{warn}")

            combo = np.hstack([yolo_panel, lift_panel])
            out_name = f"{Path(name).stem}_det{li:02d}_c{cid}.jpg"
            cv2.imwrite(str(out_dir / out_name), combo)
            records.append(
                {
                    "frame": name,
                    "det_index": li,
                    "class_id": int(cid),
                    "n_sfm_inliers": n_in,
                    "iou_reproj": iou,
                    "lifted": obj is not None,
                    "drew_reproj": drew,
                    "half_extents_m": obj.half_extents.tolist() if obj is not None else None,
                    "many_inliers_warning": n_in > 400,
                }
            )
        written += 1

    lifted = [r["iou_reproj"] for r in records if r["lifted"]]
    summary = {
        "frames_debugged": written,
        "detections": len(records),
        "median_iou": float(np.median(lifted)) if lifted else None,
        "many_inliers_count": sum(1 for r in records if r.get("many_inliers_warning")),
        "records": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir
