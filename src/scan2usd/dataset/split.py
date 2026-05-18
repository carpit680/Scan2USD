from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FRAME_STEM_RE = re.compile(r"^frame_(\d+)$", re.IGNORECASE)


def frame_label_key(frames_root: Path, image_path: Path) -> str:
    """
    Stable basename for YOLO label files under ``labels_*``, unique for nested frames.

    Examples: ``frames/a.jpg`` → ``a``; ``frames/s1/b.jpg`` → ``s1_b``.
    """
    rel = image_path.resolve().relative_to(frames_root.resolve())
    return rel.with_suffix("").as_posix().replace("/", "_").replace("\\", "_")


def label_key_from_colmap_image_name(image_name: str) -> str:
    """Match ``frame_label_key`` for a COLMAP ``images.txt`` image path (may contain ``/``)."""
    flat = Path(str(image_name)).as_posix().replace("\\", "/").replace("/", "_")
    return Path(flat).stem


def _frame_index_from_stem(stem: str) -> int | None:
    m = _FRAME_STEM_RE.match(stem)
    return int(m.group(1)) if m else None


def resolve_label_path(labels_dir: Path, image_name: str) -> Path:
    """
    Path to the YOLO label file for a COLMAP / Nerfstudio image name.

    ``ns-process-data`` renames ``frame_000000.jpg`` → ``frame_00001.jpg`` (1-based,
    often 5-digit padding). Labels from ``workspace/frames`` keep the preprocess stem;
    this resolves both naming schemes.
    """
    key = label_key_from_colmap_image_name(image_name)
    direct = labels_dir / f"{key}.txt"
    if direct.exists():
        return direct
    idx = _frame_index_from_stem(key)
    if idx is not None and idx >= 1:
        alt = labels_dir / f"frame_{idx - 1:06d}.txt"
        if alt.exists():
            return alt
    return direct


@dataclass(frozen=True)
class ImageRecord:
    """One training image with optional session id for leakage-safe splits."""

    path: Path
    session: str = "default"
    stem: str | None = None
    label_key: str | None = None

    def __post_init__(self) -> None:
        if self.stem is None:
            object.__setattr__(self, "stem", self.path.stem)
        if self.label_key is None:
            object.__setattr__(self, "label_key", self.stem)


def assign_train_val(
    records: list[ImageRecord],
    *,
    strategy: Literal["session", "random_frame"],
    val_sessions: list[str] | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """
    Split images into train/val.

    **session (recommended):** whole sessions go to val if listed in ``val_sessions``,
    or if ``val_sessions`` is empty, sessions are assigned to val by hash bucketing
    so the same session list is stable across runs.

    **random_frame:** deterministic random split by frame (weaker; may leak correlated
    frames from the same trajectory).
    """
    val_sessions = val_sessions or []
    if not records:
        return [], []

    if strategy == "random_frame":
        rng = random.Random(seed)
        idx = list(range(len(records)))
        rng.shuffle(idx)
        n_val = max(1, int(len(records) * val_ratio))
        val_set = {idx[i] for i in range(n_val)}
        train = [records[i] for i in range(len(records)) if i not in val_set]
        val = [records[i] for i in range(len(records)) if i in val_set]
        return train, val

    # session-based: whole sessions to train or val
    sessions = sorted({r.session for r in records})
    if val_sessions:
        val_sess = set(val_sessions)
        train = [r for r in records if r.session not in val_sess]
        val = [r for r in records if r.session in val_sess]
    elif len(sessions) == 1:
        # Flat capture (one session): split by frame hash, not whole session → val
        train, val = _split_records_by_frame_hash(records, val_ratio=val_ratio, seed=seed)
    else:
        rng = random.Random(seed)
        rng.shuffle(sessions)
        n_val_sess = max(1, int(len(sessions) * val_ratio))
        val_sess = set(sessions[:n_val_sess])
        train = [r for r in records if r.session not in val_sess]
        val = [r for r in records if r.session in val_sess]

    if not train or not val:
        return assign_train_val(
            records,
            strategy="random_frame",
            val_ratio=val_ratio,
            seed=seed,
        )
    return train, val


def _split_records_by_frame_hash(
    records: list[ImageRecord],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Stable per-frame val bucketing when all frames share one session."""
    train: list[ImageRecord] = []
    val: list[ImageRecord] = []
    for r in records:
        key = r.label_key or r.stem or str(r.path)
        if hash_split_session(key, val_ratio, seed):
            val.append(r)
        else:
            train.append(r)
    return train, val


def records_from_frame_dir(
    frames_dir: Path,
    *,
    session_from_parent: bool = True,
) -> list[ImageRecord]:
    """Build records from a flat or session-named subdirectory tree."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    records: list[ImageRecord] = []
    for p in sorted(frames_dir.rglob("*")):
        if p.suffix not in exts:
            continue
        if session_from_parent and p.parent != frames_dir:
            session = p.parent.name
        else:
            session = stable_session_from_path(p, frames_dir)
        lk = frame_label_key(frames_dir, p)
        records.append(ImageRecord(path=p, session=session, label_key=lk))
    return records


def stable_session_from_path(path: Path, root: Path) -> str:
    """Use relative parent path as session name for stable splits."""
    try:
        rel = path.parent.relative_to(root)
    except ValueError:
        return "default"
    if str(rel) in (".", ""):
        return "default"
    return str(rel)


def session_from_filename(stem: str, *, delimiter: str = "__") -> str:
    """Parse ``session__frame`` style stems."""
    if delimiter in stem:
        return stem.split(delimiter, 1)[0]
    return "default"


def hash_split_session(session: str, val_ratio: float, seed: int) -> bool:
    """Deterministic val assignment for a session string."""
    h = hashlib.sha256(f"{seed}:{session}".encode()).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return v < val_ratio
