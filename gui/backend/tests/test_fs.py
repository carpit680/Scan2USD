"""Filesystem browse allowlist tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.state import project_state


def test_fs_browse_home():
    project_state.set_project(
        Path(__file__).resolve(),  # may not be yaml; just set cwd
        cwd=Path.home(),
    )
    # require_config not needed for browse
    project_state.config_path = None
    project_state.cwd = Path.home().resolve()

    client = TestClient(create_app())
    r = client.get("/api/fs/roots")
    assert r.status_code == 200
    roots = r.json()["roots"]
    assert roots

    home = str(Path.home().resolve())
    r2 = client.get("/api/fs/browse", params={"path": home, "kind": "any"})
    assert r2.status_code == 200
    body = r2.json()
    assert body["path"] == home
    assert isinstance(body["entries"], list)
