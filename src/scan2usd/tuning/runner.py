"""Auto-tuner: tune config → rebuild/re-export USD → score in Isaac → retune.

Two nested loops, ordered by cost:

- **Cheap loop** — splat-cleanup thresholds. Re-cleans from
  ``environment_splat_raw.usd`` (no retraining), re-packages, renders held-out
  views in Isaac, and scores. Minutes per trial.
- **Retrain loop** (opt-in, ``max_retrain_trials > 0``) — 3DGRUT training
  parameters. Each retrain re-exports the environment ParticleField (hours on
  GPU), snapshots the splat into the trial directory, then re-runs the cheap
  loop on top of the new model.

All state lives in ``workspace/tuning/trials.json`` (resumable — completed
parameter combinations are skipped). The winner can be promoted to a
``<config>_tuned.yaml`` next to the input config.
"""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from scan2usd.config import SceneConfig
from scan2usd.eval.scene_quality import build_scene_quality_report
from scan2usd.tuning.store import TrialRecord, TrialStore

DEFAULT_CHEAP_SPACE: dict[str, list[Any]] = {
    "outlier_std": [2.0, 3.0, 4.0, 6.0],
    "min_opacity": [0.005, 0.01, 0.05],
}
DEFAULT_RETRAIN_SPACE: dict[str, list[Any]] = {
    "grut_max_iterations": [30000, 50000],
    "densify_end_fraction": [0.3, 0.5],
}


def grid(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Deterministic cross-product over {param: [values…]}."""
    keys = sorted(space)
    for combo in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, combo))


def apply_cheap_params(cfg: SceneConfig, params: dict[str, Any]) -> None:
    cleanup = cfg.reconstruction.splat_cleanup
    cleanup.enabled = True
    if "outlier_std" in params:
        cleanup.outlier_std = float(params["outlier_std"])
    if "min_opacity" in params:
        cleanup.min_opacity = float(params["min_opacity"])
    if "max_scale" in params:
        value = params["max_scale"]
        cleanup.max_scale = None if value is None else float(value)


def apply_retrain_params(cfg: SceneConfig, params: dict[str, Any]) -> None:
    recon = cfg.reconstruction
    iterations = int(params.get("grut_max_iterations", recon.grut_max_iterations))
    recon.grut_max_iterations = iterations
    overrides = [
        item
        for item in recon.grut_overrides
        if not item.startswith(
            ("scheduler.positions.max_steps=", "strategy.densify.end_iteration=")
        )
    ]
    overrides.append(f"scheduler.positions.max_steps={iterations}")
    fraction = params.get("densify_end_fraction")
    if fraction is not None:
        overrides.append(f"strategy.densify.end_iteration={int(iterations * float(fraction))}")
    if "lambda_ssim" in params:
        overrides = [o for o in overrides if not o.startswith("loss.lambda_ssim=")]
        overrides.append(f"loss.lambda_ssim={float(params['lambda_ssim'])}")
    recon.grut_overrides = overrides


class SceneTuner:
    """
    Drives trials against one workspace.

    The four pipeline actions are injectable for tests; the defaults call the
    real orchestrator / Isaac renderer.
    """

    def __init__(
        self,
        cfg: SceneConfig,
        config_path: Path,
        *,
        clean_fn: Callable[[SceneConfig], None] | None = None,
        package_fn: Callable[[SceneConfig], None] | None = None,
        render_fn: Callable[[SceneConfig, Path], None] | None = None,
        retrain_fn: Callable[[SceneConfig], None] | None = None,
        compute_lpips: bool = True,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.tuning_root = cfg.workspace_dir / "tuning"
        self.store = TrialStore(self.tuning_root / "trials.json")
        self.compute_lpips = compute_lpips
        self.log = log
        self._clean = clean_fn or self._default_clean
        self._package = package_fn or self._default_package
        self._render = render_fn or self._default_render
        self._retrain = retrain_fn or self._default_retrain

    # --- default pipeline actions -------------------------------------------------

    def _orchestrator(self):
        from scan2usd.pipeline.manifest import SceneManifest
        from scan2usd.pipeline.orchestrator import PipelineOrchestrator

        manifest = SceneManifest.load(self.cfg.workspace_dir / "scene_manifest.json")
        return PipelineOrchestrator(
            self.cfg, self.config_path, build_mode=manifest.build_mode
        )

    def _default_clean(self, cfg: SceneConfig) -> None:
        self._orchestrator().cleanup_splat(force=True)

    def _default_package(self, cfg: SceneConfig) -> None:
        from scan2usd.pipeline.manifest import SceneManifest
        from scan2usd.usd.package import build_usd_package

        manifest_path = cfg.workspace_dir / "scene_manifest.json"
        manifest = SceneManifest.load(manifest_path)
        build_usd_package(cfg, manifest)
        manifest.save(manifest_path)

    def _default_render(self, cfg: SceneConfig, output_dir: Path) -> None:
        from scan2usd.pipeline.manifest import SceneManifest
        from scan2usd.reconstruction.external_cli import (
            ExternalToolAdapter,
            resolve_external_command,
        )

        manifest_path = cfg.workspace_dir / "scene_manifest.json"
        manifest = SceneManifest.load(manifest_path)
        root_artifact = manifest.artifact("root_usd")
        if root_artifact is None:
            raise RuntimeError("No root_usd artifact; package before rendering")
        prefix = resolve_external_command(
            cfg, "isaac_python", default="python.sh", required=True
        )
        assert prefix is not None
        script = Path(__file__).resolve().parents[3] / "tools" / "isaac" / "render_heldout.py"
        ExternalToolAdapter("isaac_python", prefix).run(
            str(script),
            "--stage",
            str(Path(root_artifact.path).resolve()),
            "--held-out",
            str((cfg.workspace_dir / "build" / "grut_dataset" / "held_out.json").resolve()),
            "--colmap-txt",
            str(cfg.colmap_txt_dir.resolve()),
            "--manifest",
            str(manifest_path.resolve()),
            "--output",
            str(output_dir.resolve()),
        )

    def _default_retrain(self, cfg: SceneConfig) -> None:
        from scan2usd.pipeline.manifest import SceneManifest
        from scan2usd.reconstruction.grut import export_environment_particlefield

        manifest_path = cfg.workspace_dir / "scene_manifest.json"
        manifest = SceneManifest.load(manifest_path)
        # Stale raw backup belongs to the previous model; the fresh export reseeds it.
        raw = cfg.workspace_dir / "build" / "visual" / "environment_splat_raw.usd"
        raw.unlink(missing_ok=True)
        export_environment_particlefield(cfg, manifest)
        manifest.save(manifest_path)

    # --- trial execution ----------------------------------------------------------

    def _score_trial(self, trial: TrialRecord, render_dir: Path) -> None:
        report = build_scene_quality_report(
            self.cfg,
            render_dir=render_dir,
            compute_lpips=self.compute_lpips,
            output_path=Path(trial.artifacts_dir or self.tuning_root) / "scene_quality.json",
        )
        trial.quality_score = report["quality_score"]
        trial.metrics = {
            "mean_psnr": report["photorealism"]["mean_psnr"],
            "mean_ssim": report["photorealism"]["mean_ssim"],
            "mean_lpips": report["photorealism"]["mean_lpips"],
            "evaluated_views": report["photorealism"]["evaluated_views"],
            "gaussian_count": report["cleanliness"].get("gaussian_count"),
        }

    def run_cheap_trial(self, params: dict[str, Any]) -> TrialRecord:
        trial_id = self.store.next_trial_id("cheap")
        trial_dir = self.tuning_root / "trials" / trial_id
        render_dir = trial_dir / "renders"
        trial = TrialRecord(
            trial_id=trial_id,
            kind="cheap",
            params=dict(params),
            render_dir=str(render_dir),
            artifacts_dir=str(trial_dir),
        )
        self.store.upsert(trial)
        self.log(f"[tune] {trial_id}: {params}")
        try:
            apply_cheap_params(self.cfg, params)
            self._clean(self.cfg)
            self._package(self.cfg)
            self._render(self.cfg, render_dir)
            self._score_trial(trial, render_dir)
            trial.status = "scored"
            self.log(f"[tune] {trial_id}: quality_score={trial.quality_score}")
        except Exception as exc:  # noqa: BLE001
            trial.status = "failed"
            trial.error = f"{type(exc).__name__}: {exc}"
            self.log(f"[tune] {trial_id} FAILED: {trial.error}")
        finally:
            from scan2usd.tuning.store import _utc_now

            trial.finished_at = _utc_now()
            self.store.upsert(trial)
        return trial

    def run_cheap_loop(self, space: dict[str, list[Any]], max_trials: int) -> None:
        done = self.store.scored_params("cheap")
        for params in grid(space):
            if sum(1 for t in self.store.trials if t.kind == "cheap" and t.status == "scored") >= max_trials:
                self.log("[tune] cheap trial budget reached")
                return
            if params in done:
                continue
            self.run_cheap_trial(params)

    def run_retrain_trial(
        self,
        params: dict[str, Any],
        *,
        cheap_space: dict[str, list[Any]],
        cheap_budget: int,
    ) -> TrialRecord:
        trial_id = self.store.next_trial_id("retrain")
        trial_dir = self.tuning_root / "trials" / trial_id
        trial = TrialRecord(
            trial_id=trial_id,
            kind="retrain",
            params=dict(params),
            artifacts_dir=str(trial_dir),
        )
        self.store.upsert(trial)
        self.log(f"[tune] {trial_id} (RETRAIN — expensive): {params}")
        try:
            apply_retrain_params(self.cfg, params)
            self._retrain(self.cfg)
            # Snapshot the trained splat so a losing later retrain can't destroy it.
            visual = self.cfg.workspace_dir / "build" / "visual"
            trial_dir.mkdir(parents=True, exist_ok=True)
            for name in ("environment_splat.usd", "environment_splat_raw.usd"):
                source = visual / name
                if source.is_file():
                    shutil.copy2(source, trial_dir / name)
            self.run_cheap_loop(cheap_space, cheap_budget)
            best = self.store.best()
            trial.quality_score = best.quality_score if best else None
            trial.status = "scored"
        except Exception as exc:  # noqa: BLE001
            trial.status = "failed"
            trial.error = f"{type(exc).__name__}: {exc}"
            self.log(f"[tune] {trial_id} FAILED: {trial.error}")
        finally:
            from scan2usd.tuning.store import _utc_now

            trial.finished_at = _utc_now()
            self.store.upsert(trial)
        return trial

    # --- promotion ----------------------------------------------------------------

    def promote_best(self) -> Path | None:
        """Write the best trial's parameters to ``<config>_tuned.yaml``."""
        best = self.store.best()
        if best is None:
            return None
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        recon = raw.setdefault("reconstruction", {})
        if best.kind == "cheap":
            cleanup = recon.setdefault("splat_cleanup", {})
            cleanup.update({k: v for k, v in best.params.items()})
            cleanup["enabled"] = True
        else:
            apply_retrain_params(self.cfg, best.params)
            recon["grut_max_iterations"] = self.cfg.reconstruction.grut_max_iterations
            recon["grut_overrides"] = list(self.cfg.reconstruction.grut_overrides)
        output = self.config_path.with_name(f"{self.config_path.stem}_tuned.yaml")
        header = (
            f"# Auto-tuned from {self.config_path.name}: best trial {best.trial_id} "
            f"(quality_score={best.quality_score}).\n"
        )
        output.write_text(
            header + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
        )
        return output


def run_tuning(
    cfg: SceneConfig,
    config_path: Path,
    *,
    max_cheap_trials: int,
    max_retrain_trials: int,
    cheap_space: dict[str, list[Any]] | None = None,
    retrain_space: dict[str, list[Any]] | None = None,
    compute_lpips: bool = True,
    promote: bool = True,
    log: Callable[[str], None] = print,
    tuner: SceneTuner | None = None,
) -> dict[str, Any]:
    """Full tuning session; returns a summary dict."""
    tuner = tuner or SceneTuner(cfg, config_path, compute_lpips=compute_lpips, log=log)
    cheap = cheap_space or DEFAULT_CHEAP_SPACE
    tuner.run_cheap_loop(cheap, max_cheap_trials)
    if max_retrain_trials > 0:
        done_retrain = tuner.store.scored_params("retrain")
        started = 0
        for params in grid(retrain_space or DEFAULT_RETRAIN_SPACE):
            if started >= max_retrain_trials:
                break
            if params in done_retrain:
                continue
            tuner.run_retrain_trial(
                params,
                cheap_space=cheap,
                cheap_budget=max_cheap_trials + (started + 1) * len(list(grid(cheap))),
            )
            started += 1
    best = tuner.store.best()
    promoted = tuner.promote_best() if promote and best else None
    return {
        "trials": len(tuner.store.trials),
        "best_trial": best.trial_id if best else None,
        "best_score": best.quality_score if best else None,
        "best_params": best.params if best else None,
        "promoted_config": str(promoted) if promoted else None,
        "trials_json": str(tuner.store.path),
    }
