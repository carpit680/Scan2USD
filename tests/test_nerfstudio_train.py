"""Tests for Nerfstudio subprocess command construction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scan2usd.config import SceneConfig
from scan2usd.reconstruction import nerfstudio as ns_mod


def test_ns_train_splatfacto_quiet_logging_by_default(tmp_path: Path) -> None:
    cfg = SceneConfig.load(Path("configs/example_scene.yaml"))
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.splatfacto = cfg.splatfacto.__class__()  # defaults for isolated test
    captured: list[list[str | Path]] = []

    def fake_run(cmd: list[str | Path], **kwargs: object) -> None:
        captured.append(cmd)

    fake_config = tmp_path / "workspace" / "ns_outputs" / "splatfacto" / "splatfacto" / "run" / "config.yml"
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.touch()

    with patch.object(ns_mod, "run_cmd", side_effect=fake_run):
        with patch.object(ns_mod, "resolve_nerfstudio_cli", return_value=["ns-train"]):
            ns_mod.ns_train_splatfacto(cfg, tmp_path / "ns_data", max_num_iterations=100)

    cmd = [str(x) for x in captured[0]]
    assert "--logging.local-writer.max-log-size" in cmd
    assert cmd[cmd.index("--logging.local-writer.max-log-size") + 1] == "0"
    assert "--vis" in cmd
    assert cmd[cmd.index("--vis") + 1] == "tensorboard"
    assert "--viewer.quit-on-train-completion" not in cmd


def test_ns_train_splatfacto_viewer_opt_in(tmp_path: Path) -> None:
    cfg = SceneConfig.load(Path("configs/example_scene.yaml"))
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.splatfacto = cfg.splatfacto.__class__()  # defaults for isolated test
    captured: list[list[str | Path]] = []

    def fake_run(cmd: list[str | Path], **kwargs: object) -> None:
        captured.append(cmd)

    fake_config = tmp_path / "workspace" / "ns_outputs" / "splatfacto" / "splatfacto" / "run" / "config.yml"
    fake_config.parent.mkdir(parents=True, exist_ok=True)
    fake_config.touch()

    with patch.object(ns_mod, "run_cmd", side_effect=fake_run):
        with patch.object(ns_mod, "resolve_nerfstudio_cli", return_value=["ns-train"]):
            ns_mod.ns_train_splatfacto(
                cfg,
                tmp_path / "ns_data",
                max_num_iterations=100,
                enable_viewer=True,
            )

    cmd = [str(x) for x in captured[0]]
    assert cmd[cmd.index("--vis") + 1] == "viewer"
    assert "--viewer.quit-on-train-completion" in cmd
