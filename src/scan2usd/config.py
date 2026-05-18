from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from scan2usd.splatfacto_config import ProcessDataConfig, SplatfactoConfig


# YAML keys whose values are filesystem paths (resolved against paths base).
# Other strings (e.g. ``external`` command names, class names) must NOT be coerced to Path.
_PATH_VALUE_KEYS: frozenset[str] = frozenset(
    {
        "video_path",
        "frames_dir",
        "workspace_dir",
        "colmap_txt_dir",
        "nerfstudio_data_dir",
        "splat_config_path",
        "renders_dir",
        "dataset_dir",
    }
)


def _coerce_path_field(val: Any, base: Path) -> Any:
    if val is None:
        return None
    if isinstance(val, Path):
        return val.expanduser().resolve()
    if not isinstance(val, str) or not val or val.startswith("${"):
        return val
    p = Path(val).expanduser()
    if not p.is_absolute():
        return (base / p).resolve()
    return p.resolve()


def _apply_path_fields(data: dict[str, Any], base: Path) -> dict[str, Any]:
    out = dict(data)
    for key in _PATH_VALUE_KEYS:
        if key not in out:
            continue
        out[key] = _coerce_path_field(out[key], base)
    return out


def _paths_base(config_path: Path, raw: dict[str, Any]) -> Path:
    """
    Base directory for resolving relative paths in YAML.

    Default: current working directory (so ``workspace/`` in a config under
    ``configs/`` resolves to ``<repo>/workspace``, not ``configs/workspace``).

    Set ``paths_relative_to: config`` to resolve relative to the YAML file's directory.
    Set ``paths_relative_to: /abs/path`` (or ``~/...``) to use an explicit base.
    """
    mode = raw.get("paths_relative_to")
    if mode == "config":
        return config_path.parent.resolve()
    if isinstance(mode, str) and mode not in ("", "cwd"):
        return Path(mode).expanduser().resolve()
    return Path.cwd().resolve()


@dataclass
class SceneConfig:
    """Per-scene pipeline configuration (YAML)."""

    name: str = "default_scene"
    classes: list[str] = field(
        default_factory=lambda: [
            "chair",
            "table",
            "door",
            "couch",
            "cabinet",
            "obstacle",
            "box",
        ]
    )
    video_path: Path | None = None
    frames_dir: Path = field(default_factory=lambda: Path("workspace/frames"))
    workspace_dir: Path = field(default_factory=lambda: Path("workspace"))
    colmap_txt_dir: Path = field(default_factory=lambda: Path("workspace/colmap_txt"))
    nerfstudio_data_dir: Path = field(default_factory=lambda: Path("workspace/ns_data"))
    splat_config_path: Path | None = None
    renders_dir: Path = field(default_factory=lambda: Path("workspace/renders"))
    dataset_dir: Path = field(default_factory=lambda: Path("workspace/dataset"))
    yolo_model: str = "yolov8n.pt"
    train_epochs: int = 50
    train_imgsz: int = 640
    train_batch: int = 8
    seed: int = 42
    split: dict[str, Any] = field(
        default_factory=lambda: {
            "strategy": "session",
            "val_sessions": [],
            "val_ratio": 0.2,
        }
    )
    pose_sampling: dict[str, Any] = field(
        default_factory=lambda: {
            "num_poses": 200,
            "position_jitter_m": 0.05,
            "height_jitter_m": 0.02,
            "max_rotation_deg": 8.0,
            "interpolation_keyframes": 8,
        }
    )
    lift: dict[str, Any] = field(
        default_factory=lambda: {
            "min_points_in_box": 8,
            "merge_center_dist_m": 0.35,
        }
    )
    external: dict[str, str] = field(
        default_factory=lambda: {
            "colmap": "colmap",
            "ns_process_data": "ns-process-data",
            "ns_train": "ns-train",
            "ns_render": "ns-render",
        }
    )
    process_data: ProcessDataConfig = field(default_factory=ProcessDataConfig)
    splatfacto: SplatfactoConfig = field(default_factory=SplatfactoConfig)

    @classmethod
    def load(cls, path: Path) -> SceneConfig:
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        data = dict(raw)
        data.pop("paths_relative_to", None)  # consumed by _paths_base, not a SceneConfig field
        base = _paths_base(path, raw)
        data = _apply_path_fields(data, base)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SceneConfig:
        def p(key: str, default: Any) -> Any:
            return data[key] if key in data else default

        return cls(
            name=str(p("name", "default_scene")),
            classes=list(p("classes", SceneConfig().classes)),
            video_path=p("video_path", None),
            frames_dir=Path(p("frames_dir", Path("workspace/frames"))),
            workspace_dir=Path(p("workspace_dir", Path("workspace"))),
            colmap_txt_dir=Path(p("colmap_txt_dir", Path("workspace/colmap_txt"))),
            nerfstudio_data_dir=Path(p("nerfstudio_data_dir", Path("workspace/ns_data"))),
            splat_config_path=p("splat_config_path", None),
            renders_dir=Path(p("renders_dir", Path("workspace/renders"))),
            dataset_dir=Path(p("dataset_dir", Path("workspace/dataset"))),
            yolo_model=str(p("yolo_model", "yolov8n.pt")),
            train_epochs=int(p("train_epochs", 50)),
            train_imgsz=int(p("train_imgsz", 640)),
            train_batch=int(p("train_batch", 8)),
            seed=int(p("seed", 42)),
            split=dict(p("split", {})),
            pose_sampling=dict(p("pose_sampling", {})),
            lift=dict(p("lift", {})),
            external=dict(p("external", SceneConfig().external)),
            process_data=ProcessDataConfig.from_dict(p("process_data", None)),
            splatfacto=SplatfactoConfig.from_dict(p("splatfacto", None)),
        )
