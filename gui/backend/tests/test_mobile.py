"""Mobile LAN upload + LAN URL helpers."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scan2usd_gui.app import create_app
from scan2usd_gui.lan import build_mobile_urls, list_lan_ipv4
from scan2usd_gui.mobile_sessions import MobileSessionStore, mobile_sessions
from scan2usd_gui.state import project_state


@pytest.fixture()
def client(tmp_path: Path):
    cfg = tmp_path / "scene.yaml"
    cfg.write_text(
        "name: mobile_test\n"
        "workspace_dir: workspace\n"
        "frames_dir: workspace/frames\n"
        "classes: [chair]\n"
    )
    (tmp_path / "workspace").mkdir()
    project_state.set_project(cfg, cwd=tmp_path)
    # Fresh session store between tests
    mobile_sessions._sessions.clear()
    mobile_sessions._by_token.clear()
    yield TestClient(create_app())


def test_build_mobile_urls_includes_localhost():
    url, urls = build_mobile_urls(port=8765, token="tok")
    assert url is not None
    assert "t=tok" in url
    assert any(u.startswith("http://127.0.0.1:8765/m?") for u in urls)
    assert list_lan_ipv4() is not None


def test_create_session_requires_project(tmp_path: Path):
    project_state.config_path = None
    project_state.cwd = tmp_path
    c = TestClient(create_app())
    r = c.post("/api/mobile/sessions")
    assert r.status_code == 400


def test_mobile_upload_happy_path(client: TestClient, tmp_path: Path):
    r = client.post("/api/mobile/sessions")
    assert r.status_code == 200
    body = r.json()
    assert body["id"]
    assert body["token"]
    assert body["qr_svg"].startswith("<svg")
    assert "8765" in body["url"] or "127.0.0.1" in body["url"]

    sid, token = body["id"], body["token"]
    page = client.get("/m", params={"t": token})
    assert page.status_code == 200
    assert "Upload room video" in page.text
    assert sid in page.text

    fake = io.BytesIO(b"\x00\x00fake-mp4-bytes")
    up = client.post(
        f"/api/mobile/sessions/{sid}/upload",
        params={"token": token},
        files={"file": ("room.mp4", fake, "video/mp4")},
    )
    assert up.status_code == 200, up.text
    saved = Path(up.json()["path"])
    assert saved.is_file()
    assert saved.suffix == ".mp4"
    assert "uploads" in str(saved)

    st = client.get(f"/api/mobile/sessions/{sid}", params={"token": token})
    assert st.status_code == 200
    assert st.json()["status"] == "completed"
    assert st.json()["path"] == str(saved)


def test_reject_non_video(client: TestClient):
    body = client.post("/api/mobile/sessions").json()
    sid, token = body["id"], body["token"]
    up = client.post(
        f"/api/mobile/sessions/{sid}/upload",
        params={"token": token},
        files={"file": ("notes.txt", io.BytesIO(b"hi"), "text/plain")},
    )
    assert up.status_code == 400


def test_expired_token_rejected(client: TestClient):
    store = MobileSessionStore()
    session = store.create()
    session.expires_at = time.time() - 10
    # inject into global store
    mobile_sessions._sessions[session.id] = session
    mobile_sessions._by_token[session.token] = session.id

    up = client.post(
        f"/api/mobile/sessions/{session.id}/upload",
        params={"token": session.token},
        files={"file": ("a.mp4", io.BytesIO(b"x"), "video/mp4")},
    )
    assert up.status_code == 410

    page = client.get("/m", params={"t": session.token})
    assert page.status_code == 200
    assert "expired" in page.text.lower() or "Expired" in page.text
