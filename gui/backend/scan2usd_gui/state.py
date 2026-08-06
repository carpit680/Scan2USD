"""Mutable application state for the active project."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from scan2usd_gui.paths import repo_root, sanitize_project_cwd


class ProjectState:
    def __init__(self) -> None:
        self._lock = Lock()
        self.config_path: Path | None = None
        # Not Path.cwd(): `make gui` launches the API from gui/backend, which
        # would make every relative workspace_dir resolve into the source tree.
        self.cwd: Path = repo_root()

    def set_project(self, config_path: Path, *, cwd: Path | None = None) -> None:
        with self._lock:
            self.config_path = config_path.resolve()
            if cwd is not None:
                self.cwd = sanitize_project_cwd(cwd)

    def require_config(self) -> Path:
        with self._lock:
            if self.config_path is None or not self.config_path.is_file():
                raise FileNotFoundError(
                    "No scene config loaded. Open a YAML project first."
                )
            return self.config_path


project_state = ProjectState()
