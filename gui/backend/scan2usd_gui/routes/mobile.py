"""Mobile LAN video upload sessions."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scan2usd_gui.lan import build_mobile_urls, list_lan_ipv4, primary_lan_ipv4
from scan2usd_gui.mobile_sessions import (
    VIDEO_EXTENSIONS,
    mobile_sessions,
    qr_svg_for_url,
)
from scan2usd_gui.state import project_state

router = APIRouter(tags=["mobile"])

_STATIC = Path(__file__).resolve().parents[1] / "static" / "mobile_upload.html"


def _api_port() -> int:
    return int(os.environ.get("SCAN2USD_GUI_PORT", "8765"))


def _upload_dest_root() -> Path:
    """Resolve workspace/uploads (or cwd/uploads), matching fs.upload_file."""
    import os as _os

    from scan2usd.config import SceneConfig

    if project_state.config_path is not None:
        prev = Path.cwd()
        try:
            _os.chdir(project_state.cwd)
            cfg = SceneConfig.load(project_state.config_path)
            return Path(cfg.workspace_dir) / "uploads"
        except Exception:  # noqa: BLE001
            return project_state.cwd / "uploads"
        finally:
            try:
                _os.chdir(prev)
            except Exception:  # noqa: BLE001
                pass
    return project_state.cwd / "uploads"


class SessionCreateResponse(BaseModel):
    id: str
    token: str
    url: str
    urls: list[str]
    lan_ips: list[str]
    expires_at: float
    qr_svg: str
    port: int


class SessionStatusResponse(BaseModel):
    id: str
    status: str
    path: str | None = None
    filename: str | None = None
    expires_at: float


@router.post("/api/mobile/sessions", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    try:
        project_state.require_config()
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

    session = mobile_sessions.create()
    port = _api_port()
    url, urls = build_mobile_urls(port=port, token=session.token)
    assert url is not None
    try:
        qr_svg = qr_svg_for_url(url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"QR generation failed: {exc}") from exc

    return SessionCreateResponse(
        id=session.id,
        token=session.token,
        url=url,
        urls=urls,
        lan_ips=list_lan_ipv4(),
        expires_at=session.expires_at,
        qr_svg=qr_svg,
        port=port,
    )


@router.get("/api/mobile/sessions/by-token", response_model=SessionStatusResponse)
def session_by_token(t: str = Query(..., min_length=8)) -> SessionStatusResponse:
    session = mobile_sessions.get_by_token(t)
    if session is None:
        raise HTTPException(404, "Unknown or expired upload link")
    snap = session.snapshot()
    if snap["status"] == "expired":
        raise HTTPException(410, "Upload link expired")
    return SessionStatusResponse(**snap)


@router.get("/api/mobile/sessions/{session_id}", response_model=SessionStatusResponse)
def get_session(
    session_id: str,
    token: str = Query(..., min_length=8),
) -> SessionStatusResponse:
    session = mobile_sessions.get(session_id)
    if session is None or session.token != token:
        raise HTTPException(404, "Session not found")
    return SessionStatusResponse(**session.snapshot())


@router.post("/api/mobile/sessions/{session_id}/upload")
async def upload_video(
    session_id: str,
    token: str = Query(..., min_length=8),
    file: UploadFile = File(...),
) -> dict:
    session = mobile_sessions.get(session_id)
    if session is None or session.token != token:
        raise HTTPException(404, "Session not found")
    if session.is_expired():
        raise HTTPException(410, "Upload link expired")
    if session.status == "completed":
        raise HTTPException(409, "This upload link was already used")

    name = Path(file.filename or "video.mp4").name
    if not name or name in {".", ".."}:
        raise HTTPException(400, "Invalid filename")
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported video type {suffix!r}. Use: {', '.join(sorted(VIDEO_EXTENSIONS))}",
        )

    content_type = (file.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/")
        or content_type in {"application/octet-stream", "application/mp4"}
    ):
        raise HTTPException(400, f"Expected a video file, got content-type {content_type!r}")

    dest_root = _upload_dest_root()
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / name
    if dest.exists():
        stem, suf = dest.stem, dest.suffix
        i = 1
        while dest.exists():
            dest = dest_root / f"{stem}_{i}{suf}"
            i += 1

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    dest.write_bytes(data)
    path_str = str(dest.resolve())
    mobile_sessions.complete(session_id, path=path_str, filename=dest.name)
    return {"path": path_str, "filename": dest.name, "status": "completed"}


@router.get("/m", response_class=HTMLResponse)
def mobile_upload_page(t: str | None = None) -> HTMLResponse:
    if not _STATIC.is_file():
        raise HTTPException(500, "Mobile upload page missing")
    html = _STATIC.read_text(encoding="utf-8")
    # Soft-validate token for UI message injection
    status = "ok"
    session_id = ""
    message = ""
    if not t:
        status = "missing"
        message = "Missing upload token. Scan the QR code from the desktop again."
    else:
        session = mobile_sessions.get_by_token(t)
        if session is None:
            status = "invalid"
            message = "Unknown upload link. Ask the desktop to show a new QR code."
        elif session.is_expired():
            status = "expired"
            message = "This upload link expired. Ask the desktop for a new QR code."
        elif session.status == "completed":
            status = "done"
            message = f"Already uploaded: {session.filename or session.path}"
            session_id = session.id
        else:
            session_id = session.id

    html = (
        html.replace("{{STATUS}}", status)
        .replace("{{MESSAGE}}", message)
        .replace("{{TOKEN}}", t or "")
        .replace("{{SESSION_ID}}", session_id)
        .replace("{{PRIMARY_IP}}", primary_lan_ipv4() or "unknown")
    )
    return HTMLResponse(html)
