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
