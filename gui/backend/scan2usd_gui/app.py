"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scan2usd_gui.routes import (
    artifacts,
    config,
    doctor,
    fs,
    guide,
    jobs,
    metric,
    mobile,
    pipeline,
    project,
    review,
    tuning,
)


# Private LAN origins (Vite on phone/desktop over Wi‑Fi)
_LAN_ORIGIN_RE = (
    r"https?://("
    r"localhost|127\.0\.0\.1|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r")(:\d+)?"
)


def create_app() -> FastAPI:
    app = FastAPI(title="Scan2USD GUI", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:8765",
            "http://localhost:8765",
        ],
        allow_origin_regex=_LAN_ORIGIN_RE,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "service": "scan2usd-gui"}

    app.include_router(project.router)
    app.include_router(fs.router)
    app.include_router(config.router)
    app.include_router(jobs.router)
    app.include_router(pipeline.router)
    app.include_router(metric.router)
    app.include_router(review.router)
    app.include_router(artifacts.router)
    app.include_router(doctor.router)
    app.include_router(guide.router)
    app.include_router(mobile.router)
    app.include_router(tuning.router)

    dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if dist.is_dir():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str = "") -> FileResponse:
            # /m is owned by mobile.router; do not let SPA swallow it if order shifts
            if full_path == "m" or full_path.startswith("m/"):
                raise HTTPException(404, "Not found")
            candidate = dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
