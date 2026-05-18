from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path
from typing import Literal

from scan2usd.config import SceneConfig

CleanTier = Literal["light", "medium", "full"]


class CleanTierEnum(str, Enum):
    light = "light"
    medium = "medium"
    full = "full"


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def targets_for_tier(cfg: SceneConfig, tier: CleanTier) -> list[Path]:
    """Return paths to remove for the given cleanup tier (files or directories)."""
    ws = cfg.workspace_dir
    paths: list[Path] = []

    if tier == "full":
        paths.append(ws)
        for p in (
            cfg.frames_dir,
            cfg.colmap_txt_dir,
            cfg.nerfstudio_data_dir,
            cfg.renders_dir,
            cfg.dataset_dir,
        ):
            if p.exists() and not _is_under(p, ws):
                paths.append(p)
        return _unique_paths(paths)

    # light + medium
    paths.extend(
        [
            ws / "dataset_real",
            ws / "dataset_synthetic",
            ws / "dataset_mixed",
            ws / "reports",
        ]
    )
    if cfg.dataset_dir.exists() and not _is_under(cfg.dataset_dir, ws):
        paths.append(cfg.dataset_dir)

    if tier == "medium":
        paths.extend(
            [
                ws / "labels_real",
                ws / "labels_synthetic",
                ws / "objects_3d.npz",
                ws / "camera_path.json",
                ws / "camera_path_sanity_real.json",
                cfg.renders_dir,
            ]
        )

    return _unique_paths(paths)


def ultralytics_run_dirs(cwd: Path | None = None) -> list[Path]:
    """``runs/`` under the current working directory (Ultralytics detect/val outputs)."""
    root = (cwd or Path.cwd()).resolve()
    runs = root / "runs"
    return [runs] if runs.exists() else []


def optional_download_artifacts(cwd: Path | None = None) -> list[Path]:
    root = (cwd or Path.cwd()).resolve()
    names = ("yolo26n.pt", "yolov8n.pt")
    return [root / n for n in names if (root / n).exists()]


def run_cleanup(
    cfg: SceneConfig,
    tier: CleanTier,
    *,
    dry_run: bool = False,
    include_ultralytics: bool = True,
    include_downloads: bool = False,
    cwd: Path | None = None,
) -> list[Path]:
    """
    Remove artifacts for ``tier``. Returns paths that were removed (or would be, if dry-run).
    """
    removed: list[Path] = []
    targets = targets_for_tier(cfg, tier)
    if include_ultralytics:
        targets.extend(ultralytics_run_dirs(cwd))
    if include_downloads:
        targets.extend(optional_download_artifacts(cwd))

    for path in targets:
        if not path.exists():
            continue
        removed.append(path)
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return removed
