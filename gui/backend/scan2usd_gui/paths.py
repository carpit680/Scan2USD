"""Where the repository lives, and what a valid project working directory is.

Two different roots are easy to confuse and were, in fact, confused:

* the **repo root** — where ``tools/`` and the packaged code live. Fixed, and
  never chosen by the user.
* the **project cwd** — what relative paths inside a scene YAML resolve
  against (``workspace_dir: workspace_bedroom``). User-selectable.

``make gui`` starts the API with ``cd gui/backend``, so ``Path.cwd()`` is a
source directory. Defaulting either root to it produced
``gui/backend/tools/isaac/view_scene.py`` (nonexistent) and would have pointed
every relative ``workspace_dir`` at ``gui/backend/`` too.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _looks_like_repo(path: Path) -> bool:
    # gui/backend also has a pyproject.toml, so require both markers.
    return (path / "tools").is_dir() and (path / "pyproject.toml").is_file()


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Directory containing ``tools/`` — the Scan2USD checkout."""
    override = os.environ.get("SCAN2USD_REPO_ROOT", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if _looks_like_repo(candidate):
            return candidate

    try:
        import scan2usd

        package = Path(scan2usd.__file__).resolve()
    except Exception:  # noqa: BLE001 - fall through to cwd search
        package = None
    if package is not None:
        for base in package.parents:
            if _looks_like_repo(base):
                return base

    here = Path(__file__).resolve()
    for base in (*here.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if _looks_like_repo(base):
            return base
    return Path.cwd().resolve()


def sanitize_project_cwd(cwd: Path | None) -> Path:
    """
    Resolve a requested project cwd, refusing source directories.

    ``gui/`` holds the frontend and API source; nothing there is a data root, so
    a cwd pointing inside it is always a stale default (or stale localStorage)
    rather than an intent. Snap those to the repo root, which is what
    ``docs/USAGE.md`` tells CLI users to run from.
    """
    root = repo_root()
    resolved = (cwd or Path.cwd()).resolve()
    gui_dir = root / "gui"
    if resolved == gui_dir or gui_dir in resolved.parents:
        return root
    return resolved
