"""In-memory mobile upload sessions + QR SVG."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any


SESSION_TTL_SEC = 30 * 60
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}


@dataclass
class MobileSession:
    id: str
    token: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    status: str = "pending"  # pending | completed | expired
    path: str | None = None
    filename: str | None = None

    def __post_init__(self) -> None:
        if not self.expires_at:
            self.expires_at = self.created_at + SESSION_TTL_SEC

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def snapshot(self) -> dict[str, Any]:
        status = self.status
        if status == "pending" and self.is_expired():
            status = "expired"
        return {
            "id": self.id,
            "status": status,
            "path": self.path,
            "filename": self.filename,
            "expires_at": self.expires_at,
        }


class MobileSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, MobileSession] = {}
        self._by_token: dict[str, str] = {}
        self._lock = threading.Lock()

    def create(self) -> MobileSession:
        self._purge()
        sid = secrets.token_hex(8)
        token = secrets.token_urlsafe(24)
        session = MobileSession(id=sid, token=token)
        with self._lock:
            self._sessions[sid] = session
            self._by_token[token] = sid
        return session

    def get(self, session_id: str) -> MobileSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_by_token(self, token: str) -> MobileSession | None:
        with self._lock:
            sid = self._by_token.get(token)
            if not sid:
                return None
            return self._sessions.get(sid)

    def complete(self, session_id: str, *, path: str, filename: str) -> MobileSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.status = "completed"
            session.path = path
            session.filename = filename
            return session

    def _purge(self) -> None:
        now = time.time()
        with self._lock:
            dead = [sid for sid, s in self._sessions.items() if now > s.expires_at + 3600]
            for sid in dead:
                s = self._sessions.pop(sid, None)
                if s:
                    self._by_token.pop(s.token, None)


mobile_sessions = MobileSessionStore()


def qr_svg_for_url(url: str, *, box_size: int = 8, border: int = 2) -> str:
    """Return an SVG QR code as a string for the given URL."""
    import qrcode
    import qrcode.image.svg

    factory = qrcode.image.svg.SvgPathImage
    img = qrcode.make(url, image_factory=factory, box_size=box_size, border=border)
    return img.to_string().decode("utf-8") if isinstance(img.to_string(), bytes) else str(img.to_string())
