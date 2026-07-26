# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scan2USD turns an RGB(-D)/LiDAR room or workcell capture into a layered OpenUSD scene for **Isaac Sim 6.x**:

```
capture → metric camera/scene registration → photoreal 3DGRUT Gaussian ParticleField (visual only)
        → dense static triangle collision geometry → reviewed, separately reconstructed rigid-object
        meshes → baked + relightable PBR materials → layered USD scene with PhysX rigid bodies + validation
```

A **legacy pipeline** (Splatfacto training → synthetic-YOLO dataset generation → A/B/C detector benchmark) still lives in the same package and CLI; it is not used to produce USD assets and is being kept around only for the existing benchmark workflow. When touching code, be clear about which of the two pipelines a module belongs to (see Architecture below).

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev,geometry,review]"
```

- `make test` / `python -m pytest -q -m "not e2e"` — unit tests (default; excludes the slow e2e marker).
- `make test-e2e` / `python -m pytest -q -m e2e` — legacy synthetic-YOLO smoke test.
- Single test: `python -m pytest tests/test_static_geometry.py::test_name -q`.
- `ruff check .` — lint (config in `pyproject.toml`, line-length 100, py310).
- `scan2usd doctor configs/example_scene.yaml` — verify COLMAP/ffmpeg/Nerfstudio/CUDA/gsplat/3DGRUT/etc. before running any pipeline command; on Ubuntu it prints one aggregated `apt install` line.
- `scan2usd --help` / `scan2usd <command> --help` for the full CLI; `Makefile` has thin wrappers (`make reconstruct`, `make build-usd`, etc.) around `configs/example_scene.yaml`.
- GUI (FastAPI + React, mirrors the whole CLI): `pip install -e gui/backend && cd gui/frontend && npm install`, then `make gui` (API `:8765` + Vite `:5173`) or `make gui-lan` for phone video upload over LAN. Backend tests: `cd gui/backend && python -m pytest -q`.

Every CLI command's first argument is a scene YAML (`configs/example_scene.yaml`, `configs/high_quality_scene.yaml`). Relative paths in that YAML resolve against **the current working directory the command is run from**, not the YAML's folder, unless the YAML sets `paths_relative_to: config`. Run pipeline commands from the repo root.

## Architecture

### Two pipelines, one config/CLI

`src/scan2usd/config.py` (`SceneConfig`) is the single YAML-backed config object both pipelines read, and `src/scan2usd/cli.py` is the single Typer app both register commands on. Sub-configs (`CaptureConfig`, `ReconstructionConfig`, `SegmentationConfig`, `GeometryConfig`, `MaterialConfig`, `PhysicsConfig`, `UsdConfig`, `QAConfig`) are only used by the hybrid USD pipeline; `process_data`/`splatfacto`/`pose_sampling`/`split`/`lift` (top-level) are legacy-pipeline knobs. `WORKSPACE_PATH_LAYOUT` in `config.py` defines the standard derived paths under `workspace_dir`; `sync_workspace_paths`/`strip_workspace_derived_paths` keep the GUI's saved YAML from pinning paths that should just inherit from `workspace_dir`.

### Hybrid USD pipeline (primary)

Orchestrated by `src/scan2usd/pipeline/orchestrator.py` (`PipelineOrchestrator`), driven by a resumable, versioned `scene_manifest.json` (`src/scan2usd/pipeline/manifest.py`, `SceneManifest`). The manifest — not any in-memory state — is the source of truth: `TransformRecord` (COLMAP→USD frames, e.g. `FRAME_COLMAP`→`FRAME_USD`), `ScaleEvidence` (is this scene actually metric?), `CaptureRecord`, and per-object `ObjectRecord` (`review_state` gates the build). `PipelineOrchestrator.run_stage(...)` checks a `ready()` predicate before re-running a stage, so `build-usd` can be called repeatedly and only does new work; it raises `ReviewRequired` (CLI exit code 2) when human approval is needed before continuing.

Build order (see `PipelineOrchestrator.build`): `align_floor` → require approved object instances (else raise `ReviewRequired` pointing at `scan2usd review`) → `visual_particlefield` (3DGRUT ParticleField export) → optional `splat_cleanup` → `static_geometry` (collision + proxy mesh) → per-approved-movable-object `object_geometry`/materials/physics → lighting → `package-usd`. Each stage is also exposed as its own CLI command (`build-visual-usd`, `cleanup-splat`, `build-static-usd`, `build-object-usd`, `build-materials-usd`, `build-lighting-usd`, `package-usd`) for debugging a single stage without rerunning the whole graph.

Key modules by stage:
- `reconstruction/grut.py` — 3DGRUT ParticleField export (visual-only Gaussians); `reconstruction/splat_cleanup.py` — stray-Gaussian removal, re-runs from `environment_splat_raw.usd` (no retrain needed).
- `geometry/floor_align.py`, `geometry/frames.py`, `geometry/metric_scale.py` — floor RANSAC (Z-up, floor at Z=0) and the COLMAP→USD similarity transform; scale stays 1.0 (non-production) until `apply-metric-scale`/`set-metric-transform` is explicitly approved by a reviewer name, recorded as `ScaleEvidence.approved=True` in the manifest.
- `geometry/static_scene.py` — dense static triangle collision + alignment/depth/matte-shadow proxy mesh.
- `segmentation/propose.py` (+ Grounding-DINO-style proposal), `segmentation/propagate.py` (SAM2 mask propagation) — object instance discovery, gated by `review/app.py` (Gradio review UI) before any object is `approved`.
- `assets/object_builder.py`, `assets/materials.py`, `assets/physics.py` — per-object mesh reconstruction, baked/PBR material variants, mass/inertia/friction estimation.
- `lighting/estimate.py` — RTX dome-light estimation.
- `usd/author.py`, `usd/package.py` — compose the layered `scene.usd` (root stage + `environment/`, `objects/<instance_id>/`, `lighting/`, `semantics.usda`) from approved manifest artifacts; `usd/validate.py` — production gates (composition, meters/Z-up/ParticleField/physics schemas, registration error, manifold/UV checks, collider complexity, background coverage, held-out PSNR/SSIM/LPIPS, Isaac headless physics smoke tests), writes `build_report.json`.

Heavy/incompatible external runtimes (3DGRUT, nvblox, OpenMVS, SAM2, object-reconstruction, Isaac's `python.sh`) are **never** imported directly — they're invoked as subprocesses configured under `external:` in the scene YAML (see `reconstruction/external_cli.py`), each with its own CLI contract documented in `docs/USAGE.md` ("External runner contracts"). Don't try to merge their Python environments into the main `scan2usd` venv.

`docs/STATUS_HYBRID_USD.md` tracks live status/pending work for the current in-progress scan (what's done, what's blocked on metric scale, etc.) — check it before assuming a feature is finished.

### Legacy synthetic-YOLO pipeline

`reconstruction/nerfstudio.py` (`ns-process-data`/`ns-train splatfacto`/`ns-render` wrappers), `reconstruction/colmap_io.py` (COLMAP TXT export), `labeling/detect.py` + `labeling/lift.py`/`labeling/obb.py` (2D YOLO pseudo-labels → merged 3D oriented boxes from SfM points), `synthetic/poses.py` + `synthetic/transforms_io.py` (novel pose sampling + camera-path JSON), `export_dataset.py` + `dataset/` (YOLO dataset tree for experiments A/real, B/synthetic, C/mixed), `eval/benchmark.py` (trains/evals YOLO per experiment, `goal_C_gt_A` = does mixing beat real-only). `viz/boxes.py` + `viz/viewer.py` back `scan2usd view`/`debug-lift`, which visualize lifted 3D boxes against the Nerfstudio viser scene (mind the ×10 viser scale factor noted in `docs/USAGE.md`).

### Coordinate frames — the recurring gotcha

Several coordinate spaces coexist and are a frequent source of bugs: raw COLMAP world space, the Nerfstudio dataparser-normalized/scaled space (with its own orient/center/scale and a further ×10 viser display scale), and the USD Z-up/meters world space produced by floor alignment + metric scale. `geometry/frames.py` defines the canonical `FRAME_COLMAP`/`FRAME_USD` constants and transform validation; `pipeline/manifest.py`'s `TransformRecord`/`ScaleEvidence` are how a transform's provenance and confidence are tracked. Do not conflate the Nerfstudio normalizing transform with the floor `T` — see `docs/USAGE.md` "Accuracy vs sharpness levers".

## Config/CLI/GUI contract

`gui/backend/scan2usd_gui/schema.py` is the GUI's declared contract for every `SceneConfig` field and CLI command; its tests assert that Typer commands and config fields stay covered, so a new CLI command or config field generally needs a corresponding schema entry to keep the GUI in sync — check there when adding either.

## Environment notes

- `setuptools` must stay `<82` (pinned in `pyproject.toml`) — PyTorch/Ultralytics conflict otherwise.
- Ultralytics wants NumPy 2.x; `pip install nerfstudio` can downgrade to NumPy 1.26 transitively. Re-run `pip install "numpy>=2,<3"` after installing Nerfstudio if YOLO/OpenCV break (see README "Python deps" section for the full story, including using a second venv for Nerfstudio if needed).
- Nerfstudio must be importable from the **same** venv as `scan2usd` (`import nerfstudio` must work), or its `ns-*` scripts on `PATH`.
