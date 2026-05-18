"""Resolve Nerfstudio / COLMAP executables for subprocess invocation."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

from scan2usd.config import SceneConfig

# Map config ``external`` keys to ``python -m`` package when console scripts are missing.
_NERFSTUDIO_PY_MODULES: dict[str, str] = {
    "ns_process_data": "nerfstudio.scripts.process_data",
    "ns_train": "nerfstudio.scripts.train",
    "ns_render": "nerfstudio.scripts.render",
    "ns_viewer": "nerfstudio.scripts.viewer.run_viewer",
}


def _configured(cfg: SceneConfig, key: str, default: str) -> str:
    ext = cfg.external or {}
    v = ext.get(key, default)
    return str(v).strip() if v is not None else default


def resolve_colmap(cfg: SceneConfig) -> str:
    """Return a ``colmap`` executable path or raise with install hints."""
    name = _configured(cfg, "colmap", "colmap")
    p = Path(name).expanduser()
    if p.is_file():
        return str(p.resolve())
    w = shutil.which(Path(name).name if "/" not in name and not p.is_absolute() else name)
    if w:
        return w
    raise FileNotFoundError(
        "COLMAP executable not found. Install COLMAP (https://colmap.github.io/) "
        "or set ``external.colmap`` in your YAML to the full path to the ``colmap`` binary."
    )


def resolve_nerfstudio_cli(cfg: SceneConfig, key: str, *, default_name: str) -> list[str]:
    """
    Return argv prefix to run a Nerfstudio CLI.

    Order: explicit file path → ``shutil.which`` → ``python -m nerfstudio.scripts.*`` if importable.
    """
    name = _configured(cfg, key, default_name)
    p = Path(name).expanduser()
    if p.is_file():
        return [str(p.resolve())]
    lookup = Path(name).name if ("/" not in name and "\\" not in name) else name
    w = shutil.which(lookup)
    if w:
        return [w]
    mod = _NERFSTUDIO_PY_MODULES.get(key)
    if mod:
        root = mod.split(".", 1)[0]
        if importlib.util.find_spec(root) is not None:
            return [sys.executable, "-m", mod]
    raise FileNotFoundError(
        f"Nerfstudio command {default_name!r} not found on PATH and the ``nerfstudio`` "
        f"package is not importable. Install Nerfstudio in this venv (``pip install nerfstudio``) "
        f"or point ``external.{key}`` at the full path to the ``{default_name}`` script."
    )
