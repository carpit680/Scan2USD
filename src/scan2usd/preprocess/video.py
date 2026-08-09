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


FEATURE_EVAL_MAX_DIM = 1280


def count_sift_features(gray: np.ndarray, detector=None, max_dim: int = FEATURE_EVAL_MAX_DIM) -> int:
    """
    Number of SIFT keypoints, measured at a fixed resolution.

    SIFT is what COLMAP matches on, so this predicts whether a frame can register
    far better than a blur metric does. A sharp photo of a blank white ceiling has
    almost no features but excellent Laplacian variance-per-pixel at low
    resolution and terrible variance at 4K, which is why blur thresholds cannot be
    carried between captures. Frames are resized to a common ``max_dim`` first so
    the count means the same thing at 720p and 4K.
    """
    height, width = gray.shape[:2]
    scale = max_dim / max(height, width)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    detector = detector or cv2.SIFT_create(nfeatures=4000)
    return len(detector.detect(gray, None))


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    blur_threshold: float = 50.0,
    min_features: int = 0,
    max_dim: int = 0,
    prefix: str = "frame",
) -> list[Path]:
    """
    Extract frames from a video file (MP4, MOV, MKV, …).

    Two independent rejection rules, both optional:

    - ``blur_threshold``: Laplacian variance. Catches motion blur, but the value
      is resolution-dependent — the same sharp frame scores 20 at 720p and 3.5 at
      4K — so a threshold tuned on one capture does not transfer to another.
    - ``min_features``: SIFT keypoints at a fixed resolution. Catches frames that
      are sharp but textureless (blank walls, ceilings) which cannot register in
      COLMAP no matter how crisp they are. Resolution-independent, so this is the
      one to prefer on mixed or high-resolution captures.
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
    rejected_blur = 0
    rejected_features = 0
    detector = cv2.SIFT_create(nfeatures=4000) if min_features > 0 else None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total or None, desc="extract_frames")
    while True:
        # Frames the stride rejects are grabbed, not retrieved. grab() advances
        # the decoder exactly as read() would — so the kept frames are identical
        # — but skips the BGR conversion of the 14-in-15 frames that were only
        # ever going to be thrown away.
        if idx % stride != 0:
            if not cap.grab():
                break
            idx += 1
            pbar.update(1)
            continue
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_threshold > 0 and variance_of_laplacian(gray) < blur_threshold:
            rejected_blur += 1
            idx += 1
            pbar.update(1)
            continue
        if min_features > 0 and count_sift_features(gray, detector) < min_features:
            rejected_features += 1
            idx += 1
            pbar.update(1)
            continue
        if max_dim > 0 and max(frame.shape[:2]) > max_dim:
            scale = max_dim / max(frame.shape[:2])
            frame = cv2.resize(
                frame,
                (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        name = f"{prefix}_{saved:06d}.jpg"
        fp = out_dir / name
        cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(fp)
        saved += 1
        idx += 1
        pbar.update(1)
        if max_frames is not None and saved >= max_frames:
            break
    pbar.close()
    cap.release()
    considered = saved + rejected_blur + rejected_features
    if rejected_blur or rejected_features:
        parts = []
        if rejected_blur:
            parts.append(f"{rejected_blur} below blur threshold {blur_threshold:g}")
        if rejected_features:
            parts.append(f"{rejected_features} under {min_features} SIFT features")
        print(
            f"[extract_frames] kept {saved}/{considered} sampled frames; dropped "
            + ", ".join(parts)
            + ". Heavy feature rejection means the camera spent time on blank "
            "walls or ceilings; those frames cannot register in COLMAP.",
            flush=True,
        )
    return paths


def keyframe_subsample(paths: list[Path], every: int) -> list[Path]:
    if every <= 1:
        return paths
    return paths[::every]


FRAME_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"})


def list_frame_images(frames_dir: Path) -> list[Path]:
    """Sorted JPEG/PNG frames in a directory (non-recursive)."""
    if not frames_dir.is_dir():
        return []
    return sorted(
        p for p in frames_dir.iterdir() if p.is_file() and p.suffix in FRAME_SUFFIXES
    )


def frames_dir_has_images(frames_dir: Path) -> bool:
    """True if directory exists and contains at least one JPEG/PNG (non-recursive)."""
    if not frames_dir.is_dir():
        return False
    return any(p.suffix in FRAME_SUFFIXES for p in frames_dir.iterdir() if p.is_file())


def clear_frame_images(frames_dir: Path) -> int:
    """Delete JPEG/PNG frames in ``frames_dir`` (non-recursive). Returns count removed."""
    removed = 0
    for path in list_frame_images(frames_dir):
        path.unlink()
        removed += 1
    return removed
