from pathlib import Path

from scan2usd.cleanup import run_cleanup, targets_for_tier
from scan2usd.config import SceneConfig


def _cfg(tmp_path: Path) -> SceneConfig:
    ws = tmp_path / "workspace"
    (ws / "frames").mkdir(parents=True)
    (ws / "dataset_real" / "images").mkdir(parents=True)
    (ws / "reports").mkdir()
    (ws / "labels_real").mkdir()
    (ws / "renders").mkdir()
    (ws / "ns_outputs").mkdir()
    return SceneConfig(
        name="t",
        classes=["chair"],
        frames_dir=ws / "frames",
        workspace_dir=ws,
        colmap_txt_dir=ws / "colmap_txt",
        nerfstudio_data_dir=ws / "ns_data",
        renders_dir=ws / "renders",
        dataset_dir=ws / "dataset",
    )


def test_targets_light_vs_medium_vs_full(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    light_names = {p.name for p in targets_for_tier(cfg, "light")}
    medium_names = {p.name for p in targets_for_tier(cfg, "medium")}
    full_targets = targets_for_tier(cfg, "full")
    assert "dataset_real" in light_names
    assert "reports" in light_names
    assert "labels_real" not in light_names
    assert "labels_real" in medium_names
    assert "renders" in medium_names
    assert full_targets == [cfg.workspace_dir]


def test_run_cleanup_light(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path)
    runs = tmp_path / "runs" / "detect"
    runs.mkdir(parents=True)
    removed = run_cleanup(cfg, "light", include_ultralytics=True)
    assert not (cfg.workspace_dir / "dataset_real").exists()
    assert not (cfg.workspace_dir / "reports").exists()
    assert (cfg.workspace_dir / "labels_real").is_dir()
    assert (cfg.workspace_dir / "frames").is_dir()
    assert not runs.exists()
    assert any("dataset_real" in str(p) for p in removed)


def test_run_cleanup_full(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_cleanup(cfg, "full", include_ultralytics=False)
    assert not cfg.workspace_dir.exists()


def test_dry_run_does_not_delete(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_cleanup(cfg, "light", dry_run=True, include_ultralytics=False)
    assert (cfg.workspace_dir / "dataset_real").is_dir()
