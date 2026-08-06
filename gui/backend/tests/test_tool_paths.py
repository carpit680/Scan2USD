"""Tool scripts and project cwd must not follow the server's launch directory.

``make gui`` starts the API with ``cd gui/backend``, so ``Path.cwd()`` there is a
source directory. Resolving either root against it asked for
``gui/backend/tools/isaac/view_scene.py`` and would have pointed every relative
``workspace_dir`` into the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from scan2usd_gui.jobs import resolve_tool_argv
from scan2usd_gui.paths import repo_root, sanitize_project_cwd
from scan2usd_gui.state import ProjectState


def test_repo_root_holds_tools_regardless_of_cwd(tmp_path):
    prev = Path.cwd()
    try:
        os.chdir(tmp_path)
        root = repo_root()
    finally:
        os.chdir(prev)
    assert (root / "tools" / "isaac" / "view_scene.py").is_file()


def test_tool_script_resolves_from_the_checkout_not_the_launch_dir():
    prev = Path.cwd()
    try:
        os.chdir(repo_root() / "gui" / "backend")
        argv = resolve_tool_argv("tool-isaac-view", {})
    finally:
        os.chdir(prev)
    script = Path(argv[1])
    assert script.is_file()
    assert "gui/backend/tools" not in str(script)


def test_gui_source_dirs_are_never_a_project_cwd():
    root = repo_root()
    assert sanitize_project_cwd(root / "gui" / "backend") == root
    assert sanitize_project_cwd(root / "gui") == root
    # A real data directory is left alone.
    assert sanitize_project_cwd(Path.home()) == Path.home().resolve()


def test_default_project_cwd_is_the_repo_root(tmp_path):
    prev = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert ProjectState().cwd == repo_root()
    finally:
        os.chdir(prev)
