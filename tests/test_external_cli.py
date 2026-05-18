"""CLI resolution for COLMAP / Nerfstudio."""

from __future__ import annotations

from pathlib import Path

import pytest

from scan2usd.config import SceneConfig
from scan2usd.reconstruction.external_cli import resolve_colmap, resolve_nerfstudio_cli


def test_resolve_colmap_explicit_file(tmp_path: Path) -> None:
    fake = tmp_path / "colmap"
    fake.write_text("#!/bin/sh\necho\n")
    fake.chmod(0o755)
    cfg = SceneConfig(external={"colmap": str(fake)})
    assert resolve_colmap(cfg) == str(fake.resolve())


def test_resolve_nerfstudio_prefers_which(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = SceneConfig()
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ns-process-data" if name == "ns-process-data" else None)
    out = resolve_nerfstudio_cli(cfg, "ns_process_data", default_name="ns-process-data")
    assert out == ["/usr/bin/ns-process-data"]


def test_resolve_nerfstudio_explicit_path(tmp_path: Path) -> None:
    script = tmp_path / "ns-process-data"
    script.write_text("#!/bin/sh\necho\n")
    script.chmod(0o755)
    cfg = SceneConfig(external={"ns_process_data": str(script)})
    assert resolve_nerfstudio_cli(cfg, "ns_process_data", default_name="ns-process-data") == [str(script.resolve())]
