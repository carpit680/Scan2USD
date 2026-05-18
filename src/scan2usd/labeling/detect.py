from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm

from scan2usd.dataset.split import frame_label_key


@dataclass
class Detection:
    class_id: int
    xyxy: tuple[float, float, float, float]


def yolo_norm_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x_c, y_c = xc * img_w, yc * img_h
    bw, bh = w * img_w, h * img_h
    x0 = x_c - bw / 2
    y0 = y_c - bh / 2
    x1 = x_c + bw / 2
    y1 = y_c + bh / 2
    return x0, y0, x1, y1


def read_yolo_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        rows.append((cid, xc, yc, w, h))
    return rows


def write_yolo_label_file(path: Path, lines: Iterable[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in lines)
    path.write_text(text + ("\n" if text else ""))


def default_coco_aliases(user_classes: list[str]) -> dict[str, str]:
    """Map user class name to COCO label string used by Ultralytics COCO-pretrained models."""
    aliases: dict[str, str] = {}
    coco_map = {
        "chair": "chair",
        "table": "dining table",
        "couch": "couch",
        "door": "door",  # not in COCO80; will be skipped unless custom weights
        "cabinet": "refrigerator",  # weak proxy; prefer custom model
        "obstacle": "backpack",  # placeholder only
        "box": "handbag",
    }
    for c in user_classes:
        if c in coco_map:
            aliases[c] = coco_map[c]
    return aliases


def run_pseudo_labeling(
    frames_dir: Path,
    labels_out: Path,
    user_classes: list[str],
    *,
    weights: str = "yolov8n.pt",
    conf: float = 0.25,
    coco_aliases: dict[str, str] | None = None,
) -> None:
    """
    Run a COCO-pretrained detector and export YOLO-format labels with **user** class indices.
    Classes without a resolvable COCO alias are skipped (see ``default_coco_aliases``).
    """
    from ultralytics import YOLO

    coco_aliases = coco_aliases or default_coco_aliases(user_classes)
    name_to_idx = {n.lower(): i for i, n in enumerate(user_classes)}
    model = YOLO(weights)

    labels_out.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p
        for p in frames_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for img_path in tqdm(images, desc="pseudo_label"):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        res = model.predict(source=im, conf=conf, verbose=False)[0]
        lines: list[tuple[int, float, float, float, float]] = []
        lk = frame_label_key(frames_dir, img_path)
        if res.boxes is None:
            write_yolo_label_file(labels_out / (lk + ".txt"), lines)
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        for box, mid in zip(xyxy, cls_ids):
            mname = str(model.names[int(mid)]).lower()
            user_name = None
            for u, alias in coco_aliases.items():
                if alias.lower() == mname:
                    user_name = u
                    break
            if user_name is None:
                continue
            if user_name.lower() not in name_to_idx:
                continue
            uid = name_to_idx[user_name.lower()]
            x0, y0, x1, y1 = map(float, box.tolist())
            xc = ((x0 + x1) / 2) / max(w, 1)
            yc = ((y0 + y1) / 2) / max(h, 1)
            bw = (x1 - x0) / max(w, 1)
            bh = (y1 - y0) / max(h, 1)
            lines.append((uid, xc, yc, bw, bh))
        write_yolo_label_file(labels_out / (lk + ".txt"), lines)
