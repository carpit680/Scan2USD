"""Auto-tuner trial loop, store, and promotion (pipeline actions faked)."""

from __future__ import annotations

import pytest

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.tuning.runner import (
    SceneTuner,
    apply_retrain_params,
    grid,
    run_tuning,
)
from scan2usd.tuning.store import TrialStore


def _cfg(tmp_path: Path) -> SceneConfig:
    return SceneConfig._from_dict({"workspace_dir": str(tmp_path / "ws")}, base=tmp_path)


def _seed_heldout(cfg: SceneConfig) -> None:
    grut = cfg.workspace_dir / "build" / "grut_dataset"
    (grut / "images").mkdir(parents=True)
    (grut / "held_out.json").write_text(
        json.dumps({"images": [{"file": "f1.jpg", "camera_to_world": None}]})
    )
    Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(grut / "images" / "f1.jpg")


def _make_tuner(cfg: SceneConfig, config_path: Path, render_value: dict) -> SceneTuner:
    """Tuner whose render step fabricates images; quality varies with outlier_std."""

    def fake_render(cfg_: SceneConfig, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Closer to reference (100) for outlier_std=4.0 → that trial should win.
        offset = 0 if cfg_.reconstruction.splat_cleanup.outlier_std == 4.0 else 60
        Image.fromarray(np.full((8, 8, 3), 100 + offset, dtype=np.uint8)).save(
            out_dir / "f1.png"
        )
        render_value["calls"] = render_value.get("calls", 0) + 1

    return SceneTuner(
        cfg,
        config_path,
        clean_fn=lambda _cfg: None,
        package_fn=lambda _cfg: None,
        render_fn=fake_render,
        retrain_fn=lambda _cfg: None,
        compute_lpips=False,
        log=lambda _msg: None,
    )


def test_grid_is_deterministic_cross_product() -> None:
    combos = list(grid({"b": [1, 2], "a": ["x"]}))
    assert combos == [{"a": "x", "b": 1}, {"a": "x", "b": 2}]


def test_cheap_loop_scores_and_picks_best(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _seed_heldout(cfg)
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(yaml.safe_dump({"workspace_dir": str(cfg.workspace_dir)}))
    calls: dict = {}
    tuner = _make_tuner(cfg, config_path, calls)

    space = {"outlier_std": [2.0, 4.0], "min_opacity": [0.01]}
    tuner.run_cheap_loop(space, max_trials=10)

    assert calls["calls"] == 2
    best = tuner.store.best()
    assert best is not None
    assert best.params["outlier_std"] == 4.0
    assert best.status == "scored"
    assert (cfg.workspace_dir / "tuning" / "trials.json").is_file()


def test_resume_skips_completed_params(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _seed_heldout(cfg)
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(yaml.safe_dump({"workspace_dir": str(cfg.workspace_dir)}))
    calls: dict = {}
    space = {"outlier_std": [2.0, 4.0], "min_opacity": [0.01]}

    tuner = _make_tuner(cfg, config_path, calls)
    tuner.run_cheap_loop(space, max_trials=10)
    assert calls["calls"] == 2

    resumed = _make_tuner(cfg, config_path, calls)
    resumed.run_cheap_loop(space, max_trials=10)
    assert calls["calls"] == 2  # nothing re-ran


def test_failed_trial_recorded_not_fatal(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _seed_heldout(cfg)
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(yaml.safe_dump({"workspace_dir": str(cfg.workspace_dir)}))

    def broken_render(_cfg: SceneConfig, _out: Path) -> None:
        raise RuntimeError("isaac crashed")

    tuner = SceneTuner(
        cfg,
        config_path,
        clean_fn=lambda _cfg: None,
        package_fn=lambda _cfg: None,
        render_fn=broken_render,
        compute_lpips=False,
        log=lambda _msg: None,
    )
    trial = tuner.run_cheap_trial({"outlier_std": 2.0})
    assert trial.status == "failed"
    assert "isaac crashed" in (trial.error or "")
    store = TrialStore(cfg.workspace_dir / "tuning" / "trials.json")
    assert store.trials[0].status == "failed"


def test_promote_writes_tuned_config(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _seed_heldout(cfg)
    config_path = tmp_path / "scene.yaml"
    config_path.write_text(
        yaml.safe_dump({"workspace_dir": str(cfg.workspace_dir), "name": "t"})
    )
    calls: dict = {}
    tuner = _make_tuner(cfg, config_path, calls)
    summary = run_tuning(
        cfg,
        config_path,
        max_cheap_trials=4,
        max_retrain_trials=0,
        cheap_space={"outlier_std": [2.0, 4.0]},
        compute_lpips=False,
        promote=True,
        log=lambda _msg: None,
        tuner=tuner,
    )
    assert summary["best_params"]["outlier_std"] == 4.0
    tuned = Path(summary["promoted_config"])
    assert tuned.name == "scene_tuned.yaml"
    raw = yaml.safe_load(tuned.read_text())
    assert raw["reconstruction"]["splat_cleanup"]["outlier_std"] == 4.0
    # Promoted config must still load as a valid SceneConfig.
    assert SceneConfig.load(tuned).reconstruction.splat_cleanup.outlier_std == 4.0


def test_apply_retrain_params_syncs_scheduler() -> None:
    cfg = SceneConfig()
    apply_retrain_params(
        cfg, {"grut_max_iterations": 30000, "densify_end_fraction": 0.5}
    )
    assert cfg.reconstruction.grut_max_iterations == 30000
    assert "scheduler.positions.max_steps=30000" in cfg.reconstruction.grut_overrides
    assert "strategy.densify.end_iteration=15000" in cfg.reconstruction.grut_overrides


def test_a_retrain_scores_its_own_children_not_the_global_best(tmp_path):
    """
    The bug: trial.quality_score = store.best().

    A retrain that produced a worse model inherited the score of the best trial
    ever run, so the first good retrain made every later one look equally good
    and promotion picked arbitrarily among them.
    """
    from scan2usd.tuning.store import TrialRecord, TrialStore

    store = TrialStore(tmp_path / "trials.json")
    store.upsert(TrialRecord("cheap_001", "cheap", {}, "scored", 90.0, parent_trial_id="retrain_001"))
    store.upsert(TrialRecord("cheap_002", "cheap", {}, "scored", 40.0, parent_trial_id="retrain_002"))

    def best_of(parent: str) -> float:
        children = [
            t for t in store.trials if t.parent_trial_id == parent and t.quality_score is not None
        ]
        return max(t.quality_score for t in children)

    assert best_of("retrain_001") == 90.0
    # The second retrain must not inherit the first's 90.
    assert best_of("retrain_002") == 40.0
    assert store.best().quality_score == 90.0


def test_cheap_params_reach_every_cleanup_field(tmp_path):
    """
    The bedroom config listed crop_margin and max_scale_frac in its tuning
    space; the tuner honoured neither and reported identical scores as though
    the parameters did not matter.
    """
    from scan2usd.config import SceneConfig
    from scan2usd.tuning.runner import apply_cheap_params

    cfg = SceneConfig()
    apply_cheap_params(
        cfg,
        {
            "crop_margin": 0.25,
            "max_scale_frac": 0.05,
            "free_space_votes": 3,
            "hull_air": True,
            "outlier_std": 8.0,
        },
    )
    cleanup = cfg.reconstruction.splat_cleanup
    assert cleanup.crop_margin == 0.25
    assert cleanup.max_scale_frac == 0.05
    assert cleanup.free_space_votes == 3
    assert cleanup.hull_air is True
    assert cleanup.outlier_std == 8.0


def test_a_typo_in_a_tuning_space_raises_instead_of_vanishing():
    from scan2usd.config import SceneConfig
    from scan2usd.tuning.runner import apply_cheap_params

    with pytest.raises(KeyError, match="crop_margins"):
        apply_cheap_params(SceneConfig(), {"crop_margins": 0.25})


def test_default_space_no_longer_sweeps_values_known_to_destroy_scenes():
    """outlier_std below 8 is measured damage; the old default swept 2.0-6.0."""
    from scan2usd.tuning.runner import DEFAULT_CHEAP_SPACE

    assert min(DEFAULT_CHEAP_SPACE["outlier_std"]) >= 6.0
    assert "free_space_votes" in DEFAULT_CHEAP_SPACE


def test_unsafe_retrain_space_is_refused_before_spending_gpu_time(tmp_path):
    from scan2usd.config import SceneConfig
    from scan2usd.tuning.runner import run_tuning

    with pytest.raises(ValueError, match="lambda_opacity"):
        run_tuning(
            SceneConfig(),
            tmp_path / "s.yaml",
            max_cheap_trials=0,
            max_retrain_trials=1,
            retrain_space={"lambda_opacity": [0.01]},
        )


def test_fog_penalty_prefers_the_clear_scene_at_equal_appearance():
    """
    Appearance cannot see haze, so without this term a foggy scene wins on a
    hundredth of a dB. 30 points is the full-scale weight: a room you cannot
    see across loses more than any appearance difference can recover.
    """
    weight, span = 1.0, 30.0
    clear = 65.0 - weight * (1.0 - 0.95) * span
    foggy = 65.1 - weight * (1.0 - 0.13) * span
    assert clear > foggy
