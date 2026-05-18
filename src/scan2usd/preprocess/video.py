from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Containers OpenCV typically reads when built with FFmpeg (includes QuickTime / .mov).
VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4",
        ".m4v",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".wmv",
        ".mpg",
        ".mpeg",
    }
)


def is_supported_video_suffix(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    blur_threshold: float = 50.0,
    prefix: str = "frame",
) -> list[Path]:
    """
    Extract frames from a video file (MP4, MOV, MKV, …); drop blurry frames
    (Laplacian variance below threshold). Requires OpenCV with FFmpeg for most
    formats, including ``.mov``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    path_s = str(video_path.resolve())
    # Prefer FFmpeg backend so MOV / H.264 in QuickTime containers work on typical Linux builds.
    cap = cv2.VideoCapture(path_s, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(path_s)
    if not cap.isOpened():
        hint = (
            f"Cannot open video: {video_path}. "
            "Install/use OpenCV built with FFmpeg; MOV and many MP4 variants need it. "
            f"Known extensions: {', '.join(sorted(VIDEO_SUFFIXES))}."
        )
        raise RuntimeError(hint)
    paths: list[Path] = []
    idx = 0
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total or None, desc="extract_frames")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            pbar.update(1)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if variance_of_laplacian(gray) < blur_threshold:
            idx += 1
            pbar.update(1)
            continue
        name = f"{prefix}_{saved:06d}.jpg"
        fp = out_dir / name
        cv2.imwrite(str(fp), frame)
        paths.append(fp)
        saved += 1
        idx += 1
        pbar.update(1)
        if max_frames is not None and saved >= max_frames:
            break
    pbar.close()
    cap.release()
    return paths


def keyframe_subsample(paths: list[Path], every: int) -> list[Path]:
    if every <= 1:
        return paths
    return paths[::every]


def frames_dir_has_images(frames_dir: Path) -> bool:
    """True if directory exists and contains at least one JPEG/PNG (non-recursive)."""
    if not frames_dir.is_dir():
        return False
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return any(p.suffix in exts for p in frames_dir.iterdir() if p.is_file())
