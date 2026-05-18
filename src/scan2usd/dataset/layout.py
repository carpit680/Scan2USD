from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml


def yolo_dir_layout(root: Path) -> dict[str, Path]:
    """Standard YOLO layout from goal.md."""
    out = {
        "root": root,
        "images_train": root / "images" / "train",
        "images_val": root / "images" / "val",
        "labels_train": root / "labels" / "train",
        "labels_val": root / "labels" / "val",
    }
    return out


def ensure_yolo_tree(root: Path) -> dict[str, Path]:
    layout = yolo_dir_layout(root)
    for k, p in layout.items():
        if k == "root":
            continue
        p.mkdir(parents=True, exist_ok=True)
    return layout


def write_data_yaml(
    root: Path,
    class_names: list[str],
    *,
    train_rel: str = "images/train",
    val_rel: str = "images/val",
) -> Path:
    """Write Ultralytics-style ``data.yaml`` pointing at this dataset root."""
    ensure_yolo_tree(root)
    data = {
        "path": str(root.resolve()),
        "train": train_rel,
        "val": val_rel,
        "names": {i: n for i, n in enumerate(class_names)},
        "nc": len(class_names),
    }
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def _clear_split_dir(img_dir: Path, lbl_dir: Path) -> None:
    for d in (img_dir, lbl_dir):
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file():
                p.unlink()


def copy_image_label_pairs(
    train_pairs: Iterable[tuple[Path, Path | None]],
    val_pairs: Iterable[tuple[Path, Path | None]],
    root: Path,
) -> None:
    """Copy (image, optional yolo label) into train/val folders."""
    import shutil

    layout = ensure_yolo_tree(root)
    _clear_split_dir(layout["images_train"], layout["labels_train"])
    _clear_split_dir(layout["images_val"], layout["labels_val"])
    for img, lbl in train_pairs:
        dst_i = layout["images_train"] / img.name
        shutil.copy2(img, dst_i)
        if lbl and lbl.exists():
            dst_l = layout["labels_train"] / (img.stem + ".txt")
            shutil.copy2(lbl, dst_l)
        else:
            dst_l = layout["labels_train"] / (img.stem + ".txt")
            dst_l.write_text("")

    for img, lbl in val_pairs:
        dst_i = layout["images_val"] / img.name
        shutil.copy2(img, dst_i)
        if lbl and lbl.exists():
            dst_l = layout["labels_val"] / (img.stem + ".txt")
            shutil.copy2(lbl, dst_l)
        else:
            dst_l = layout["labels_val"] / (img.stem + ".txt")
            dst_l.write_text("")
