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

# Everything below 8.0 is measured damage: on the kitchen 50k model outlier_std
# 4.0 cut 56% of the Gaussians and tore holes in floors and walls, scoring 66.7
# against 74.1 at 8.0. The old default swept 2.0-6.0, i.e. only values the
# project had already shown to be harmful.
DEFAULT_CHEAP_SPACE: dict[str, list[Any]] = {
    "outlier_std": [6.0, 8.0],
    "min_opacity": [0.005, 0.01],
    "free_space_votes": [0, 3, 10],
    "crop_margin": [0.25, 0.5],
}
DEFAULT_RETRAIN_SPACE: dict[str, list[Any]] = {
    "grut_max_iterations": [30000, 50000],
    "densify_end_fraction": [0.3, 0.5],
}
# Trials whose parameters are known to destroy a scene are refused before the
# GPU time is spent: lambda_opacity 0.01 collapsed the bedroom from 2.9M
# Gaussians to 8,576 while reporting a healthy loss throughout.
UNSAFE_RETRAIN: dict[str, tuple[float, str]] = {
    "lambda_opacity": (
        0.005,
        "collapsed the bedroom to 8,576 Gaussians from 2.9M — it drives "
        "opacities under the density-prune threshold as fast as densification "
        "creates them",
    ),
}


def grid(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Deterministic cross-product over {param: [values…]}."""
    keys = sorted(space)
    for combo in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, combo))


def apply_cheap_params(cfg: SceneConfig, params: dict[str, Any]) -> None:
    """
    Apply any SplatCleanupConfig field named in a trial.

    Previously only outlier_std, min_opacity and max_scale were honoured, and
    anything else was silently ignored — so the bedroom config listed
    crop_margin and max_scale_frac in its tuning space and the tuner quietly
    swept neither, reporting identical scores as if the parameter did not
    matter. Unknown names now raise rather than disappear.
    """
    import dataclasses

    cleanup = cfg.reconstruction.splat_cleanup
    cleanup.enabled = True
    known = {f.name: f for f in dataclasses.fields(cleanup)}
    for name, value in params.items():
        field = known.get(name)
        if field is None:
            raise KeyError(
                f"{name!r} is not a splat_cleanup parameter. Available: "
                + ", ".join(sorted(known))
            )
        if value is None:
            setattr(cleanup, name, None)
            continue
        annotation = str(field.type)
        if "bool" in annotation:
            setattr(cleanup, name, bool(value))
        elif "int" in annotation and "float" not in annotation:
            setattr(cleanup, name, int(value))
        else:
            setattr(cleanup, name, float(value))


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
        fog_weight: float = 1.0,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.tuning_root = cfg.workspace_dir / "tuning"
        self.store = TrialStore(self.tuning_root / "trials.json")
        self.compute_lpips = compute_lpips
        self.fog_weight = float(fog_weight)
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
        score = report["quality_score"]
        fog = self._fog_metrics()
        # Held-out appearance is structurally blind to interior haze: it renders
        # roughly the pixels the surface behind it would. Without this term the
        # tuner cannot tell a clear room from a foggy one, and will happily pick
        # the foggy one for a hundredth of a dB.
        transmittance = (fog or {}).get("transmittance_across_room")
        penalty = 0.0
        if score is not None and transmittance is not None:
            penalty = self.fog_weight * (1.0 - float(transmittance)) * 30.0
            score = score - penalty
        trial.quality_score = score
        trial.metrics = {
            "mean_psnr": report["photorealism"]["mean_psnr"],
            "mean_ssim": report["photorealism"]["mean_ssim"],
            "mean_lpips": report["photorealism"]["mean_lpips"],
            "evaluated_views": report["photorealism"]["evaluated_views"],
            "gaussian_count": report["cleanliness"].get("gaussian_count"),
            "appearance_score": report["quality_score"],
            "transmittance_across_room": transmittance,
            "fog_penalty": round(penalty, 3),
        }

    def _fog_metrics(self) -> dict[str, Any] | None:
        """Fog from the cleanup report this trial just wrote — no extra work."""
        import json as _json

        path = (
            Path(self.cfg.workspace_dir)
            / "build"
            / "visual"
            / "splat_cleanup_report.json"
        )
        try:
            if path.is_file():
                return _json.loads(path.read_text(encoding="utf-8")).get("fog_metrics")
        except (OSError, ValueError):
            return None
        return None

    def run_cheap_trial(
        self, params: dict[str, Any], *, parent_trial_id: str | None = None
    ) -> TrialRecord:
        trial_id = self.store.next_trial_id("cheap")
        trial_dir = self.tuning_root / "trials" / trial_id
        render_dir = trial_dir / "renders"
        trial = TrialRecord(
            trial_id=trial_id,
            kind="cheap",
            params=dict(params),
            render_dir=str(render_dir),
            artifacts_dir=str(trial_dir),
            parent_trial_id=parent_trial_id,
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

    def run_cheap_loop(
        self,
        space: dict[str, list[Any]],
        max_trials: int,
        *,
        parent_trial_id: str | None = None,
    ) -> None:
        done = self.store.scored_params("cheap")
        for params in grid(space):
            if sum(1 for t in self.store.trials if t.kind == "cheap" and t.status == "scored") >= max_trials:
                self.log("[tune] cheap trial budget reached")
                return
            if params in done:
                continue
            self.run_cheap_trial(params, parent_trial_id=parent_trial_id)

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
            self.run_cheap_loop(cheap_space, cheap_budget, parent_trial_id=trial_id)
            # This retrain's own best cleanup, not the best trial ever run.
            children = [
                t
                for t in self.store.trials
                if t.parent_trial_id == trial_id and t.quality_score is not None
            ]
            trial.quality_score = (
                max(t.quality_score for t in children) if children else None
            )
            trial.metrics = {"cheap_children": len(children)}
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
        # This file is a full snapshot of the source config, not a patch, so it
        # rots as the source moves on. A kitchen_scene_tuned.yaml written in July
        # still carried the CPU COLMAP path and none of the halo filters added a
        # few hours later; running it would have been a 69x slower reconstruction
        # with worse cleanup, for a win of 0.014 quality points. Record what it
        # was cut from so that is checkable.
        import hashlib

        source_text = self.config_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(source_text.encode()).hexdigest()[:12]
        header = (
            f"# Auto-tuned from {self.config_path.name}: best trial {best.trial_id} "
            f"(quality_score={best.quality_score}).\n"
            f"# Snapshot of {self.config_path.name}@{digest} — everything outside\n"
            "# the tuned parameters is frozen at that revision and will not pick up\n"
            "# later edits. Re-run `scan2usd tune` after changing the source, or\n"
            "# copy the winning parameters back and delete this file.\n"
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
    fog_weight: float = 1.0,
    promote: bool = True,
    log: Callable[[str], None] = print,
    tuner: SceneTuner | None = None,
) -> dict[str, Any]:
    """Full tuning session; returns a summary dict."""
    tuner = tuner or SceneTuner(
        cfg, config_path, compute_lpips=compute_lpips, fog_weight=fog_weight, log=log
    )
    cheap = cheap_space or DEFAULT_CHEAP_SPACE
    for space in (cheap, retrain_space or DEFAULT_RETRAIN_SPACE):
        for name, (limit, why) in UNSAFE_RETRAIN.items():
            values = [v for v in space.get(name, []) if v is not None and float(v) > limit]
            if values:
                raise ValueError(
                    f"tuning space has {name}={values}, above the safe limit "
                    f"{limit}: {why}. Refusing to spend GPU time on it."
                )
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
