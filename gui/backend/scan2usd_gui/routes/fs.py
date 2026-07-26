"""Local filesystem browse / upload for path pickers (allowlisted roots)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from scan2usd.config import SceneConfig

from scan2usd_gui.state import project_state

router = APIRouter(prefix="/api/fs", tags=["fs"])

_UNSAFE = re.compile(r"[\x00]")


def _allowed_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        try:
            resolved = p.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen:
            return
        if resolved.exists():
            seen.add(key)
            roots.append(resolved)

    add(project_state.cwd)
    add(Path.cwd())
    add(Path.home())
    # Common media mounts on Linux
    for extra in ("/media", "/mnt", "/data"):
        add(Path(extra))

    if project_state.config_path is not None:
        add(project_state.config_path.parent)
        try:
            prev = Path.cwd()
            os.chdir(project_state.cwd)
            cfg = SceneConfig.load(project_state.config_path)
            add(Path(cfg.workspace_dir))
            add(Path(cfg.frames_dir))
            if cfg.video_path:
                add(Path(cfg.video_path).parent)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                os.chdir(prev)
            except Exception:  # noqa: BLE001
                pass

    # Repo root guess: parent of cwd if it contains configs/
    for candidate in (project_state.cwd, Path.cwd()):
        if (candidate / "configs").is_dir():
            add(candidate)
        if (candidate.parent / "configs").is_dir():
            add(candidate.parent)

    return roots


def _is_under_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in _allowed_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
        # also allow the root itself
    for root in _allowed_roots():
        if resolved == root:
            return True
    return False


@router.get("/roots")
def list_roots() -> dict:
    return {
        "roots": [
            {"path": str(r), "label": r.name or str(r), "is_home": r == Path.home().resolve()}
            for r in _allowed_roots()
        ]
    }


@router.get("/browse")
def browse(
    path: str | None = Query(None),
    kind: str = Query("any"),  # file | dir | any
    ext: str | None = Query(None, description="Comma-separated extensions, e.g. .mp4,.mov,.yaml"),
) -> dict:
    roots = _allowed_roots()
    if not roots:
        raise HTTPException(500, "No browsable roots available")

    if path:
        if _UNSAFE.search(path):
            raise HTTPException(400, "Invalid path")
        target = Path(path).expanduser()
        try:
            target = target.resolve()
        except OSError as exc:
            raise HTTPException(400, str(exc)) from exc
    else:
        target = roots[0]

    if not _is_under_allowed(target):
        raise HTTPException(403, "Path is outside allowlisted directories")

    if not target.exists():
        raise HTTPException(404, f"Not found: {target}")
    if not target.is_dir():
        raise HTTPException(400, "Browse path must be a directory")

    extensions: set[str] = set()
    if ext:
        for part in ext.split(","):
            part = part.strip().lower()
            if not part:
                continue
            if not part.startswith("."):
                part = f".{part}"
            extensions.add(part)

    entries: list[dict] = []
    try:
        children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc

    for child in children[:3000]:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if kind == "dir" and not is_dir:
            continue
        if kind == "file" and is_dir:
            # still show dirs so user can navigate
            pass
        if not is_dir and extensions and child.suffix.lower() not in extensions:
            continue
        if kind == "file" and not is_dir and extensions and child.suffix.lower() not in extensions:
            continue
        entries.append(
            {
                "name": child.name,
                "path": str(child.resolve()),
                "is_dir": is_dir,
                "suffix": child.suffix.lower() if not is_dir else "",
                "size": child.stat().st_size if not is_dir else None,
            }
        )

    # parent if still under allowlist
    parent = target.parent
    parent_path = str(parent.resolve()) if _is_under_allowed(parent) else None

    return {
        "path": str(target),
        "parent": parent_path,
        "kind": kind,
        "ext": ext,
        "entries": entries,
        "roots": [str(r) for r in roots],
    }


class UploadResponse(BaseModel):
    path: str
    filename: str


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Save an uploaded file under workspace/uploads (or cwd/uploads)."""
    name = Path(file.filename or "upload.bin").name
    if not name or name in {".", ".."}:
        raise HTTPException(400, "Invalid filename")

    dest_root: Path
    if project_state.config_path is not None:
        try:
            prev = Path.cwd()
            os.chdir(project_state.cwd)
            cfg = SceneConfig.load(project_state.config_path)
            dest_root = Path(cfg.workspace_dir) / "uploads"
        except Exception:  # noqa: BLE001
            dest_root = project_state.cwd / "uploads"
        finally:
            try:
                os.chdir(prev)
            except Exception:  # noqa: BLE001
                pass
    else:
        dest_root = project_state.cwd / "uploads"

    dest_root.mkdir(parents=True, exist_ok=True)
    if not _is_under_allowed(dest_root) and not _is_under_allowed(dest_root.parent):
        # workspace may be newly created — allow if under cwd
        try:
            dest_root.resolve().relative_to(project_state.cwd.resolve())
        except ValueError as exc:
            raise HTTPException(403, "Upload destination not allowed") from exc

    dest = dest_root / name
    # avoid overwrite clobber: unique name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 1
        while dest.exists():
            dest = dest_root / f"{stem}_{i}{suffix}"
            i += 1

    data = await file.read()
    dest.write_bytes(data)
    return UploadResponse(path=str(dest.resolve()), filename=dest.name)
