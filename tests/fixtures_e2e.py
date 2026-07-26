"""Minimal on-disk workspace for end-to-end pipeline tests (no ``ns-process-data``)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def write_minimal_scene_yaml(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "paths_relative_to: config",
                "name: e2e",
                "classes:",
                "  - chair",
                "  - table",
                "video_path: null",
                "workspace_dir: e2e_workspace",
                "frames_dir: e2e_workspace/frames",
                "colmap_txt_dir: e2e_workspace/colmap_txt",
                "nerfstudio_data_dir: e2e_workspace/ns_data",
                "splat_config_path: null",
                "renders_dir: e2e_workspace/renders",
                "dataset_dir: e2e_workspace/dataset",
                "yolo_model: yolov8n.pt",
                "train_epochs: 1",
                "train_imgsz: 320",
                "train_batch: 2",
                "seed: 42",
                "split:",
                "  strategy: random_frame",
                "  val_ratio: 0.25",
                "pose_sampling:",
                "  num_poses: 4",
                "  position_jitter_m: 0.02",
                "  height_jitter_m: 0.01",
                "  max_rotation_deg: 4.0",
                "  interpolation_keyframes: 4",
                "lift:",
                "  min_points_in_box: 2",
                "  merge_center_dist_m: 2.0",
                "external:",
                "  colmap: colmap",
                "  ns_process_data: ns-process-data",
                "  ns_train: ns-train",
                "  ns_render: ns-render",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_e2e_workspace(root: Path) -> Path:
    """
    Create ``root/e2e_workspace`` with frames, COLMAP TXT, labels, ``ns_data`` (transforms + images).

    COLMAP geometry: one PINHOLE camera; identity world-to-cam; points near (0,0,5).
    """
    ws = root / "e2e_workspace"
    frames = ws / "frames"
    colmap_txt = ws / "colmap_txt"
    labels = ws / "labels_real"
    ns_data = ws / "ns_data"
    ns_img = ns_data / "images"
    renders = ws / "renders"
    for d in (frames, colmap_txt, labels, ns_img, renders):
        d.mkdir(parents=True, exist_ok=True)

    W, H = 640, 480
    n_im = 10
    for i in range(n_im):
        arr = np.full((H, W, 3), (i * 20) % 256, dtype=np.uint8)
        fp = frames / f"e{i:02d}.jpg"
        im = Image.fromarray(arr)
        im.save(fp, quality=90)

        shutil.copy2(fp, ns_img / f"e{i:02d}.jpg")
        # YOLO: class 0 (chair), centered box covering principal point
        (labels / f"e{i:02d}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    (colmap_txt / "cameras.txt").write_text(
        f"#\n1 PINHOLE {W} {H} 500.0 500.0 320.0 240.0\n", encoding="utf-8"
    )

    img_lines: list[str] = ["#"]
    for i in range(1, n_im + 1):
        name = f"e{i - 1:02d}.jpg"
        img_lines.append(
            f"{i} 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 {name}\n" + "0 0 -1\n"
        )
    (colmap_txt / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    pts: list[str] = ["#"]
    pid = 1
    rng = np.random.default_rng(0)
    for _ in range(25):
        jitter = rng.normal(0, 0.02, size=3)
        xyz = np.array([0.0, 0.0, 5.0]) + jitter
        pts.append(
            f"{pid} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} 200 200 200 0.01 1 1 1\n",
        )
        pid += 1
    (colmap_txt / "points3D.txt").write_text("\n".join(pts) + "\n", encoding="utf-8")
    xyz_rows = [" ".join(line.split()[1:4]) for line in pts[1:]]
    (ns_data / "sparse_pc.ply").write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(xyz_rows)}",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                *xyz_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )

    eye = np.eye(4).tolist()
    frames_meta = []
    for i in range(n_im):
        frames_meta.append(
            {
                "file_path": f"./images/e{i:02d}.jpg",
                "transform_matrix": eye,
            }
        )
    transforms = {
        "w": W,
        "h": H,
        "fl_x": 500.0,
        "fl_y": 500.0,
        "cx": 320.0,
        "cy": 240.0,
        "camera_model": "PINHOLE",
        "frames": frames_meta,
    }
    (ns_data / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")

    for j in range(4):
        Image.new("RGB", (W, H), color=(j * 40, 100, 80)).save(renders / f"r{j:02d}.png")

    return ws
