"""In-app user guide."""

from __future__ import annotations

from fastapi import APIRouter

from scan2usd_gui.guide import get_guide

router = APIRouter(prefix="/api/guide", tags=["guide"])


@router.get("")
def guide() -> dict:
    return get_guide()
