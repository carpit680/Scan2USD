from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from scan2usd.config import SceneConfig
from scan2usd.dataset.layout import copy_image_label_pairs, ensure_yolo_tree, write_data_yaml
from scan2usd.dataset.split import assign_train_val, records_from_frame_dir, resolve_label_path
from scan2usd.labeling.detect import write_yolo_label_file
from scan2usd.labeling.lift import Object3D, intrinsics_from_transforms_meta, project_aabb_to_yolo_line_c2w


def _copy_image_label_dir(src_img_dir: Path, src_lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    if not src_img_dir.exists():
        return
    for p in sorted(src_img_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        shutil.copy2(p, dst_img_dir / p.name)
        lp = src_lbl_dir / (p.stem + ".txt")
        dst_l = dst_lbl_dir / (p.stem + ".txt")
        if lp.exists():
            shutil.copy2(lp, dst_l)
        else:
            dst_l.write_text("")


def _image_dir_for_dataset(cfg: SceneConfig) -> Path:
    """Prefer Nerfstudio images when present so stems match ``labels_real`` from ``label``."""
    ns_images = cfg.nerfstudio_data_dir / "images"
    if ns_images.is_dir() and any(ns_images.glob("*")):
        return ns_images
    return cfg.frames_dir


def build_real_yolo_dataset(
    cfg: SceneConfig,
    *,
    labels_dir: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    """
    Materialize ``images/{train,val}`` + ``labels/*`` + ``data.yaml`` from frames on disk.
    """
    root = output_root or (cfg.workspace_dir / "dataset_real")
    labels_dir = labels_dir or (cfg.workspace_dir / "labels_real")
    images_dir = _image_dir_for_dataset(cfg)
    records = records_from_frame_dir(images_dir)
    strat = str(cfg.split.get("strategy", "session"))
    val_sessions = list(cfg.split.get("val_sessions") or [])
    val_ratio = float(cfg.split.get("val_ratio", 0.2))
    train_r, val_r = assign_train_val(
        records,
        strategy="session" if strat == "session" else "random_frame",
        val_sessions=val_sessions,
        val_ratio=val_ratio,
        seed=cfg.seed,
    )

    def pairs(recs):
        return [(r.path, resolve_label_path(labels_dir, r.path.name)) for r in recs]

    copy_image_label_pairs(pairs(train_r), pairs(val_r), root)
    write_data_yaml(root, cfg.classes)
    return root / "data.yaml"


def copy_real_val_into(root: Path, real_root: Path) -> None:
    """Attach real val split for evaluation (Experiments B/C)."""
    layout = ensure_yolo_tree(root)
    _copy_image_label_dir(
        real_root / "images" / "val",
        real_root / "labels" / "val",
        layout["images_val"],
        layout["labels_val"],
    )


def write_synthetic_labels(
    objects: list[Object3D],
    c2w_list: list[np.ndarray],
    meta: dict,
    width: int,
    height: int,
    labels_dir: Path,
) -> None:
    intr = intrinsics_from_transforms_meta(meta, width, height)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for i, c2w in enumerate(c2w_list):
        lines: list[tuple[int, float, float, float, float]] = []
        for obj in objects:
            line = project_aabb_to_yolo_line_c2w(obj.bbox, obj.class_id, c2w, intr)
            if line is not None:
                lines.append(line)
        write_yolo_label_file(labels_dir / f"synth_{i:06d}.txt", lines)


def materialize_synthetic_train_split(
    cfg: SceneConfig,
    renders_dir: Path,
    labels_dir: Path,
    *,
    real_root_for_val: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    """
    Copy rendered RGB into ``images/train`` with matching YOLO labels.
    If ``real_root_for_val`` is set, also copy real val images/labels for held-out evaluation.
    """
    root = output_root or (cfg.workspace_dir / "dataset_synthetic")
    layout = ensure_yolo_tree(root)
    images = sorted(
        [p for p in renders_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )
    for idx, src in enumerate(images):
        dst = layout["images_train"] / f"synth_{idx:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        lbl_src = labels_dir / f"synth_{idx:06d}.txt"
        lbl_dst = layout["labels_train"] / f"synth_{idx:06d}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")
    if real_root_for_val is not None:
        copy_real_val_into(root, real_root_for_val)
    write_data_yaml(root, cfg.classes)
    return root / "data.yaml"


def build_mixed_dataset(
    cfg: SceneConfig,
    real_root: Path,
    synthetic_root: Path,
    output_root: Path | None = None,
) -> Path:
    """Real train + synthetic train; val is real val only."""
    root = output_root or (cfg.workspace_dir / "dataset_mixed")
    layout = ensure_yolo_tree(root)
    _copy_image_label_dir(
        real_root / "images" / "train",
        real_root / "labels" / "train",
        layout["images_train"],
        layout["labels_train"],
    )
    _copy_image_label_dir(
        synthetic_root / "images" / "train",
        synthetic_root / "labels" / "train",
        layout["images_train"],
        layout["labels_train"],
    )
    _copy_image_label_dir(
        real_root / "images" / "val",
        real_root / "labels" / "val",
        layout["images_val"],
        layout["labels_val"],
    )
    write_data_yaml(root, cfg.classes)
    return root / "data.yaml"
