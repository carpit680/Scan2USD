"""Bridge between GUI API and Scan2USD SceneConfig / YAML."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from scan2usd.config import SceneConfig, sync_workspace_paths

from scan2usd_gui.state import project_state


def _path_to_str(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _path_to_str(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_path_to_str(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _path_to_str(asdict(value))
    return value


def scene_config_to_dict(cfg: SceneConfig) -> dict[str, Any]:
    """Serialize SceneConfig to a JSON-friendly nested dict."""
    return {
        "name": cfg.name,
        "classes": list(cfg.classes),
        "video_path": None if cfg.video_path is None else str(cfg.video_path),
        "frames_dir": str(cfg.frames_dir),
        "workspace_dir": str(cfg.workspace_dir),
        "colmap_txt_dir": str(cfg.colmap_txt_dir),
        "nerfstudio_data_dir": str(cfg.nerfstudio_data_dir),
        "splat_config_path": (
            None if cfg.splat_config_path is None else str(cfg.splat_config_path)
        ),
        "renders_dir": str(cfg.renders_dir),
        "dataset_dir": str(cfg.dataset_dir),
        "yolo_model": cfg.yolo_model,
        "train_epochs": cfg.train_epochs,
        "train_imgsz": cfg.train_imgsz,
        "train_batch": cfg.train_batch,
        "seed": cfg.seed,
        "split": dict(cfg.split),
        "pose_sampling": dict(cfg.pose_sampling),
        "lift": dict(cfg.lift),
        "external": dict(cfg.external),
        "process_data": _path_to_str(cfg.process_data),
        "splatfacto": _path_to_str(cfg.splatfacto),
        "capture": _path_to_str(cfg.capture),
        "reconstruction": _path_to_str(cfg.reconstruction),
        "segmentation": _path_to_str(cfg.segmentation),
        "geometry": _path_to_str(cfg.geometry),
        "materials": _path_to_str(cfg.materials),
        "physics": _path_to_str(cfg.physics),
        "usd": _path_to_str(cfg.usd),
        "qa": _path_to_str(cfg.qa),
    }


def load_raw_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def load_project(config_path: Path, *, cwd: Path | None = None) -> dict[str, Any]:
    path = config_path.resolve()
    work_cwd = (cwd or Path.cwd()).resolve()
    # SceneConfig resolves relative paths against cwd unless paths_relative_to is set
    import os

    prev = Path.cwd()
    try:
        os.chdir(work_cwd)
        cfg = SceneConfig.load(path)
        raw = load_raw_yaml(path)
        yaml_text = path.read_text()
    finally:
        os.chdir(prev)
    project_state.set_project(path, cwd=work_cwd)
    return {
        "config_path": str(path),
        "cwd": str(work_cwd),
        "paths_relative_to": raw.get("paths_relative_to", "cwd"),
        "config": scene_config_to_dict(cfg),
        "raw": raw,
        "yaml_text": yaml_text,
    }


def save_raw_yaml(
    path: Path,
    data: dict[str, Any],
    *,
    previous_workspace_dir: str | None = None,
) -> dict[str, Any]:
    """Write YAML then reload through SceneConfig for validation."""
    normalized = sync_workspace_paths(
        data, previous_workspace_dir=previous_workspace_dir
    )
    text = yaml.safe_dump(normalized, sort_keys=False, default_flow_style=False)
    path.write_text(text)
    return load_project(path, cwd=project_state.cwd)


def save_yaml_text(path: Path, text: str) -> dict[str, Any]:
    """Parse YAML text, validate via SceneConfig, and write the original text."""
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError("YAML must be a mapping (top-level object)")
    # Validate without writing first
    import os
    import tempfile

    prev = Path.cwd()
    try:
        os.chdir(project_state.cwd)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        try:
            SceneConfig.load(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    finally:
        os.chdir(prev)
    path.write_text(text)
    return load_project(path, cwd=project_state.cwd)


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _display_path(path: Path | None, *, cwd: Path) -> str | None:
    """Prefer a path relative to the project cwd when it lives under that tree."""
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (cwd / resolved).resolve()
    else:
        resolved = resolved.resolve()
    try:
        return str(resolved.relative_to(cwd.resolve()))
    except ValueError:
        return str(resolved)


def workspace_paths(cfg: SceneConfig, *, cwd: Path | None = None) -> dict[str, str | None]:
    """Resolved scene paths for Commands / tools defaults (keys used by ``default_from``)."""
    work = (cwd or project_state.cwd).resolve()
    ws = Path(cfg.workspace_dir)
    usd_dir = Path(cfg.usd.output_dir) if cfg.usd.output_dir else ws / "usd"
    masks = (
        Path(cfg.segmentation.masks_dir)
        if cfg.segmentation.masks_dir
        else ws / "masks"
    )
    visual = ws / "build" / "visual"
    env_splat = visual / "environment_splat.usd"
    # Prefer manifest artifact when present (package may relocate the USD).
    manifest_path = ws / "scene_manifest.json"
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            arts = raw.get("artifacts") if isinstance(raw, dict) else None
            if isinstance(arts, dict):
                splat_art = arts.get("environment_splat")
                if isinstance(splat_art, dict) and splat_art.get("path"):
                    env_splat = Path(str(splat_art["path"]))
                root_art = arts.get("root_usd")
                if isinstance(root_art, dict) and root_art.get("path"):
                    root_from_manifest = Path(str(root_art["path"]))
                else:
                    root_from_manifest = None
            else:
                root_from_manifest = None
        except (OSError, json.JSONDecodeError, TypeError):
            root_from_manifest = None
    else:
        root_from_manifest = None

    root_usd = root_from_manifest or (usd_dir / cfg.usd.root_filename)
    splat_parent = env_splat.parent
    env_splat_raw = splat_parent / "environment_splat_raw.usd"
    # Prefer raw backup when it exists so re-cleanup is re-runnable from unfiltered export.
    cleanup_input = env_splat_raw if env_splat_raw.is_file() else env_splat

    def d(p: Path | None) -> str | None:
        return _display_path(p, cwd=work)

    return {
        "workspace_dir": d(ws),
        "frames_dir": d(Path(cfg.frames_dir)),
        "colmap_txt_dir": d(Path(cfg.colmap_txt_dir)),
        "nerfstudio_data_dir": d(Path(cfg.nerfstudio_data_dir)),
        "renders_dir": d(Path(cfg.renders_dir)),
        "dataset_dir": d(Path(cfg.dataset_dir)),
        "masks_dir": d(masks),
        "usd_dir": d(usd_dir),
        "root_usd": d(root_usd),
        "environment_splat": d(env_splat),
        "environment_splat_raw": d(env_splat_raw),
        "environment_splat_cleanup_input": d(cleanup_input),
        "splat_cleanup_report": d(splat_parent / "splat_cleanup_report.json"),
        "proposals_json": d(ws / "build" / "segmentation" / "proposals.json"),
        "segmentation_dir": d(ws / "build" / "segmentation"),
        "colmap_to_usd_floor": d(ws / "colmap_to_usd_floor.json"),
        "colmap_to_usd_metric": d(ws / "colmap_to_usd_metric.json"),
        "objects_3d": d(ws / "objects_3d.npz"),
        "video_path": d(Path(cfg.video_path) if cfg.video_path else None),
        "validate_report": d(ws / "build" / "validate_report.json"),
        "isaac_validate_report": d(ws / "build" / "isaac_validate_report.json"),
    }


def workspace_summary(cfg: SceneConfig, *, cwd: Path | None = None) -> dict[str, Any]:
    work = (cwd or project_state.cwd).resolve()
    paths = workspace_paths(cfg, cwd=work)
    ws = Path(cfg.workspace_dir)
    manifest = ws / "scene_manifest.json"
    state = ws / "build" / "pipeline_state.json"
    return {
        "workspace_dir": paths["workspace_dir"],
        "exists": ws.is_dir(),
        "has_manifest": manifest.is_file(),
        "has_pipeline_state": state.is_file(),
        "usd_dir": paths["usd_dir"],
        "frames_dir": paths["frames_dir"],
        "masks_dir": paths["masks_dir"],
        "paths": paths,
    }


def _config_value_by_path(cfg: SceneConfig, dotted: str) -> Any:
    """Read a dotted path from SceneConfig (e.g. reconstruction.splat_cleanup.outlier_std)."""
    cur: Any = cfg
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    if isinstance(cur, Path):
        return str(cur)
    return cur


def apply_command_path_defaults(
    command_id: str,
    options: dict[str, Any],
    cfg: SceneConfig,
    *,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Fill empty options from workspace paths / scene config when schema declares defaults."""
    from scan2usd_gui.schema import COMMAND_DEFS, TOOL_DEFS

    cmd = next(
        (c for c in (*COMMAND_DEFS, *TOOL_DEFS) if c["id"] == command_id),
        None,
    )
    if cmd is None:
        return dict(options)
    paths = workspace_paths(cfg, cwd=cwd)
    out = dict(options)
    for opt in cmd.get("options") or []:
        key = opt.get("config_path") or opt.get("id")
        if not key:
            continue
        cur = out.get(key)
        if cur is not None and cur != "":
            continue
        default_from = opt.get("default_from")
        if default_from:
            suggested = paths.get(str(default_from))
            if suggested:
                out[key] = suggested
                continue
        config_from = opt.get("config_default_from")
        if config_from:
            suggested = _config_value_by_path(cfg, str(config_from))
            if suggested is not None and suggested != "":
                out[key] = suggested
    return out


def object_to_dict(obj: Any) -> dict[str, Any]:
    """Serialize a manifest ObjectRecord-like object."""
    if hasattr(obj, "__dict__"):
        data = dict(obj.__dict__)
    else:
        data = dict(obj)
    return json.loads(json.dumps(data, default=str))
