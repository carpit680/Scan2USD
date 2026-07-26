"""Mutable application state for the active project."""

from __future__ import annotations

from pathlib import Path
from threading import Lock


class ProjectState:
    def __init__(self) -> None:
        self._lock = Lock()
        self.config_path: Path | None = None
        self.cwd: Path = Path.cwd().resolve()

    def set_project(self, config_path: Path, *, cwd: Path | None = None) -> None:
        with self._lock:
            self.config_path = config_path.resolve()
            if cwd is not None:
                self.cwd = cwd.resolve()

    def require_config(self) -> Path:
        with self._lock:
            if self.config_path is None or not self.config_path.is_file():
                raise FileNotFoundError(
                    "No scene config loaded. Open a YAML project first."
                )
            return self.config_path


project_state = ProjectState()
