"""Resolve Nerfstudio / COLMAP executables for subprocess invocation."""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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


def _resolve_executable(name: str) -> str | None:
    path = Path(name).expanduser()
    if path.is_file():
        # Preserve venv interpreter symlinks. Resolving ``.venv/bin/python`` to
        # ``/usr/bin/python`` discards the virtual environment's site-packages.
        return str(path.absolute())
    lookup = Path(name).name if ("/" not in name and "\\" not in name) else name
    return shutil.which(lookup)


def resolve_external_command(
    cfg: SceneConfig,
    key: str,
    *,
    default: str,
    python_module: str | None = None,
    required: bool = True,
) -> list[str] | None:
    """
    Resolve a configured external command without invoking a shell.

    Values may be a binary path/name or a quoted argv prefix such as
    ``docker run --rm image``. The first token must resolve to an executable.
    """
    configured = _configured(cfg, key, default)
    if not configured:
        if required:
            raise FileNotFoundError(f"external.{key} is not configured")
        return None
    argv = shlex.split(configured)
    if not argv:
        if required:
            raise FileNotFoundError(f"external.{key} is empty")
        return None
    executable = _resolve_executable(argv[0])
    if executable:
        return [executable, *argv[1:]]
    if python_module and importlib.util.find_spec(python_module.split(".", 1)[0]) is not None:
        return [sys.executable, "-m", python_module]
    if not required:
        return None
    raise FileNotFoundError(
        f"External tool {key!r} could not be resolved from {configured!r}. "
        f"Install it or set external.{key} to an executable path/argv prefix."
    )


@dataclass
class ExternalToolAdapter:
    """Pinned external runtime boundary with deterministic argv/env handling."""

    name: str
    argv_prefix: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    def command(self, *args: str | Path) -> list[str]:
        return [*self.argv_prefix, *(str(arg) for arg in args)]

    def run(
        self,
        *args: str | Path,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        merged = os.environ.copy()
        merged.update(self.env)
        return subprocess.run(
            self.command(*args),
            check=check,
            cwd=str(self.cwd) if self.cwd else None,
            env=merged,
            text=True,
            capture_output=capture_output,
        )

    def version(self, *version_args: str) -> str:
        args = version_args or ("--version",)
        result = self.run(*args, check=False, capture_output=True)
        output = (result.stdout or result.stderr or "").strip()
        return output.splitlines()[0] if output else f"exit={result.returncode}"


def external_tool(
    cfg: SceneConfig,
    key: str,
    *,
    default: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    required: bool = True,
) -> ExternalToolAdapter | None:
    prefix = resolve_external_command(cfg, key, default=default, required=required)
    if prefix is None:
        return None
    return ExternalToolAdapter(key, prefix, cwd=cwd, env=env or {})


def resolve_colmap(cfg: SceneConfig) -> str:
    """Return a ``colmap`` executable path or raise with install hints."""
    name = _configured(cfg, "colmap", "colmap")
    resolved = _resolve_executable(name)
    if resolved:
        return resolved
    raise FileNotFoundError(
        "COLMAP executable not found. Install COLMAP (https://colmap.github.io/) "
        "or set ``external.colmap`` in your YAML to the full path to the ``colmap`` binary."
    )


def resolve_nerfstudio_cli(cfg: SceneConfig, key: str, *, default_name: str) -> list[str]:
    """
    Return argv prefix to run a Nerfstudio CLI.

    Order: explicit file path → ``shutil.which`` → ``python -m nerfstudio.scripts.*`` if importable.
    """
    mod = _NERFSTUDIO_PY_MODULES.get(key)
    try:
        resolved = resolve_external_command(
            cfg,
            key,
            default=default_name,
            python_module=mod,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Nerfstudio command {default_name!r} not found on PATH and the ``nerfstudio`` "
            f"package is not importable. Install Nerfstudio in this venv "
            f"(``pip install nerfstudio``) or point ``external.{key}`` at the full path "
            f"to the ``{default_name}`` script."
        ) from exc
    assert resolved is not None
    return resolved
