"""Nerfstudio subprocess environment helpers."""

from __future__ import annotations

from scan2usd.reconstruction.nerfstudio import _apply_nerfstudio_runtime_env


def test_apply_nerfstudio_runtime_env_torch_load() -> None:
    env: dict[str, str] = {}
    _apply_nerfstudio_runtime_env(env)
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"

    env2 = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "0"}
    _apply_nerfstudio_runtime_env(env2)
    assert env2["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "0"


def test_process_data_passes_configured_colmap(monkeypatch, tmp_path):
    """external.colmap must reach ns-process-data, or a CUDA build goes unused."""
    from scan2usd.config import SceneConfig
    from scan2usd.reconstruction import nerfstudio

    captured = {}
    monkeypatch.setattr(nerfstudio, "run_cmd", lambda cmd, **kw: captured.setdefault("cmd", cmd))
    monkeypatch.setattr(
        nerfstudio, "resolve_nerfstudio_cli", lambda *a, **k: ["ns-process-data"]
    )
    fake = tmp_path / "colmap-cuda"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    cfg = SceneConfig()
    cfg.external["colmap"] = str(fake)
    nerfstudio.ns_process_data_images(cfg, tmp_path / "frames", tmp_path / "out")

    cmd = [str(c) for c in captured["cmd"]]
    assert "--colmap-cmd" in cmd
    assert cmd[cmd.index("--colmap-cmd") + 1] == str(fake)
