# Scan2USD — Usage manual

Primary pipeline: **room/workcell capture → Gaussian ParticleField visuals + solid geometry → reviewed rigid objects → layered Isaac Sim USD**.

The former Splatfacto → synthetic-YOLO benchmark remains documented as a legacy workflow.

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Installation](#installation)
3. [Scene configuration (YAML)](#scene-configuration-yaml)
4. [Path resolution](#path-resolution)
5. [Workspace layout](#workspace-layout)
6. [Commands reference](#commands-reference)
7. [Makefile shortcuts](#makefile-shortcuts)
8. [Experiments A / B / C](#experiments-a--b--c)
9. [Nerfstudio tools (direct)](#nerfstudio-tools-direct)
10. [Environment variables](#environment-variables)
11. [Troubleshooting](#troubleshooting)
12. [Capture guidance](#capture-guidance)

---

## Hybrid Scan-to-USD workflow

### What the export contains

`scan2usd` composes a root `scene.usd` from independent layers:

```text
workspace/usd/
  scene.usd                         # root stage, variants, PhysicsScene
  environment/
    splat.usd                       # 3DGRUT standard ParticleField; visual only
    collision.usdc                  # invisible static triangle collision
    proxy.usdc                      # alignment/depth/matte-shadow proxy
  objects/<instance_id>/
    geometry.usdc                   # local render mesh
    collision.usdc                  # convex-decomposition source
    materials.usda                  # baked and PBR variants
    physics.usda                    # mass, inertia, friction, restitution
    asset.usd                       # reusable object interface
  lighting/lighting.usda
  semantics.usda
  scene_manifest.json
  build_report.json
```

The Gaussian environment is intentionally not a collider. Omniverse RTX renders the standard OpenUSD `UsdVol.ParticleField3DGaussianSplat` alongside polygonal objects; separate meshes provide PhysX contacts, depth/shadow proxies, and object manipulation.

### Input quality tiers

- **RGB-D/LiDAR:** production path. A configured nvblox runner fuses metric static geometry and emits `depth_validation.json`.
- **RGB-only:** COLMAP + OpenMVS path. After reconstruct, `align-floor` (also run automatically by `build-usd`) fits a floor plane and stores a rigid COLMAP→USD transform (Z-up, floor at Z=0, unit scale). In Isaac, that Z=0 plane is the ground (gravity −Z); there is no separate Kit ground collider. Production remains blocked until `apply-metric-scale` or `set-metric-transform` approves true metric scale.
- **One-pass capture:** preview quality when an object hides unobserved background.
- **Clean plate/object detail passes:** production path for ghost-free removal and complete object geometry.

A generic RGB video cannot guarantee metric, watertight geometry. The manifest records confidence and the production builder fails instead of silently inventing physics.

### External runner contracts

Heavy runtimes stay isolated and are configured under `external`:

- `grut_root` points to NVIDIA 3DGRUT v1.1+; `grut_python` runs `train.py`.
- `sam2_runner` receives `--images`, `--proposals`, `--output`, `--mask-format foreground-white`; it writes `<output>/<instance_id>/<frame>.png`.
- `object_reconstruction_runner` receives `--images`, `--masks`, `--colmap`, `--instance-id`, `--output-mesh`, `--texture-resolution`, and optional `--detail-capture`; it writes a textured OBJ (or one PLY fallback).
- `nvblox` receives `--input-manifest`, `--output-mesh`, `--validation-report`, `--voxel-size`, sensor paths, and `--static-only`.
- OpenMVS uses `InterfaceCOLMAP`, `DensifyPointCloud`, `ReconstructMesh`, and `RefineMesh`.
- `isaac_python` points to Isaac Sim 6.x `python.sh`.

Run `scan2usd doctor configs/example_scene.yaml` before building.

### Build and review

```bash
scan2usd reconstruct configs/example_scene.yaml
scan2usd init-usd configs/example_scene.yaml --mode production
scan2usd segment-usd configs/example_scene.yaml
scan2usd review configs/example_scene.yaml

# RGB-only: estimate floor plane → Z-up alignment (scale stays 1 until a metric anchor).
scan2usd align-floor configs/example_scene.yaml

# Remove stray Gaussians from the environment ParticleField (no retrain).
# Primary knob: reconstruction.splat_cleanup.outlier_std (higher = keep more floaters).
# Re-runs from environment_splat_raw.usd so you can tune and re-package quickly.
scan2usd cleanup-splat configs/example_scene.yaml
# scan2usd cleanup-splat configs/example_scene.yaml --outlier-std 3.0

# Metric scale: measure a known length, then compose scale onto the floor transform.
scan2usd apply-metric-scale configs/example_scene.yaml \
  --known-length-m 0.91 --source-length 2.45 --reviewer NAME
# Or: scan2usd apply-metric-scale configs/example_scene.yaml --meters-per-unit 0.37 --reviewer NAME
# Or approve a full 4×4 JSON directly:
# scan2usd set-metric-transform configs/example_scene.yaml transform.json --reviewer NAME

# After packaging, Isaac ground is USD Z=0 (from align-floor). The viewer draws a
# translucent non-colliding marker there by default (`--no-ground-marker` to hide).
# workspace/isaac_env/bin/python tools/isaac/view_scene.py --stage workspace/usd/scene.usd

# Resumable; pauses at required human gates.
scan2usd build-usd configs/example_scene.yaml
scan2usd review configs/example_scene.yaml
scan2usd build-usd configs/example_scene.yaml

# Compare Isaac renders at held-out cameras and run headless physics tests.
scan2usd validate-usd configs/example_scene.yaml \
  --held-out-renders workspace/isaac_heldout_renders
```

Focused commands (`build-visual-usd`, `cleanup-splat`, `build-static-usd`, `build-object-usd`, `build-materials-usd`, `build-lighting-usd`, and `package-usd`) expose individual stages for debugging.

Live status (done / pending / next focus) for the hybrid kitchen scene is in
[`docs/STATUS_HYBRID_USD.md`](STATUS_HYBRID_USD.md).

### Environment splat quality (3DGRUT)

**Config tiers:** use **`configs/high_quality_scene.yaml`** for final hybrid USD builds
(50k iters, full-res COLMAP via `process_data.num_downscales: 1`, densify/SSIM/SH
overrides). It is a drop-in replacement for `configs/example_scene.yaml` (same workspace
paths). Use **`configs/example_scene.yaml`** for day-to-day work (currently also 50k-capable
but `num_downscales: 2` to save disk/VRAM). For quick pipeline checks, set
**`grut_max_iterations: 5000`** (preview — often soft / under-densified).

- **`grut_max_iterations: 5000`** — fast preview; often soft / under-densified.
- **`grut_max_iterations: 30000`** — production-ish sharpness (much slower). Match
  `scheduler.positions.max_steps` via `grut_overrides` so the LR schedule finishes.
- **`configs/high_quality_scene.yaml`** — 50k + held-out 5% + splat cleanup; see file for
  full `grut_overrides`. Rebuild with:
  `scan2usd build-visual-usd configs/high_quality_scene.yaml`.
- Approved **movable** objects are **masked out** of the environment ParticleField on
  purpose (hybrid design). They appear as separate object meshes. Holes behind them
  need `capture.clean_plate_dir`, not longer training.
- Extra Hydra knobs (densify, SH, loss) go in `reconstruction.grut_overrides`:

```yaml
reconstruction:
  grut_max_iterations: 30000
  held_out_ratio: 0.05
  grut_overrides:
    - scheduler.positions.max_steps=30000
    - strategy.densify.end_iteration=15000
```

Rebuild the visual layer after changing these:

```bash
scan2usd build-visual-usd configs/example_scene.yaml
scan2usd package-usd configs/example_scene.yaml
```

### Accuracy vs sharpness levers

**Accuracy** (pose, floor/gravity, metric sizes, collision vs visual):

1. `apply-metric-scale` with a measured length (biggest Isaac physics gap).
2. Keep `align-floor`; re-run if COLMAP changes. Do not mix Nerfstudio normalizing
   transforms with the floor `T`.
3. Dense static mesh baked once under the current metric `T` (identity parent xform).
4. Clean plate + detail object captures so manipulables match size/pose.
5. `validate-usd` once metric; tighten `qa.max_registration_error_m`.

**Sharpness** (ParticleField look):

1. Train with `configs/high_quality_scene.yaml` (SH ramp every 500 steps +
   `num_downscales: 1` — may need re-preprocess for full-res COLMAP).
2. More iterations / densify help up to ~50k; gains diminish after that.
3. Capture quality (less blur, more overlap, better lighting).
4. `splat_cleanup.outlier_std`: higher keeps fine haze, more floaters; lower cleans
   more aggressively.
5. Movables are masked from the env splat — sharpen those via object meshes, not
   longer environment training.

### Review gates

The local `scan2usd review` app:

1. shows propagated instance masks over real frames;
2. allows corrected foreground-white masks to be imported;
3. confirms class, movable/static status, background coverage, and physical template;
4. approves generated geometry, PBR estimates, physical properties, and lighting.

Production consumers only read the approved `scene_manifest.json`.

### Appearance and lighting

- `look=baked` uses capture-matched emissive texture. It is useful for static appearance comparison, but captured shadows move with the object.
- `look=pbr` uses a conservative de-lit base-color estimate plus normal, roughness, and metalness maps. This is the default for moving objects and must be reviewed.
- The ParticleField contains captured radiance. RTX can render it with mesh shadows/reflections, but it is not a fully relightable PBR surface.

### Quality metrics and auto-tuning

The end-metric for photorealism is **Isaac renders at held-out capture cameras vs
the real frames**:

```bash
# Render held-out views headlessly in Isaac and score PSNR/SSIM/LPIPS.
scan2usd render-heldout configs/golden_scene.yaml
# Re-aggregate scene_quality.json without re-rendering.
scan2usd quality-report configs/golden_scene.yaml
```

Camera poses/intrinsics come from the COLMAP TXT model (same raw COLMAP frame as
the exported splat) mapped through the manifest COLMAP→USD transform — never from
`transforms.json` (different frame). Results land in
`workspace/usd/scene_quality.json` (`quality_score` 0–100: SSIM 45% / LPIPS 35% /
PSNR 20%, with penalties for registration error and failed validation gates) and
`photorealism_report.json`. `validate-usd` picks up `build/heldout_renders`
automatically.

**Auto-tuner** (`scan2usd tune`, or the GUI **Tuning** page):

```bash
scan2usd tune configs/golden_scene.yaml                      # cheap loop only
scan2usd tune configs/golden_scene.yaml --retrain-trials 2   # + 3DGRUT retrains
```

- *Cheap loop*: sweeps `splat_cleanup` thresholds by re-cleaning from
  `environment_splat_raw.usd` (no retraining) → package → Isaac render → score.
- *Retrain loop* (opt-in): sweeps `grut_max_iterations` / densify schedule; hours
  of GPU per trial; each trained splat is snapshotted under `workspace/tuning/trials/`.
- Resumable via `workspace/tuning/trials.json`; the winner is written to
  `<config>_tuned.yaml` (`--no-promote` to skip). Budgets/spaces live under the
  `tuning:` YAML section.

`configs/golden_scene.yaml` is the annotated best-practices profile and the
recommended tuner seed. For maximum photorealism with no manipulable objects,
it sets `segmentation.allow_no_objects: true` (nothing is masked out of the splat).

### Validation

`validate-usd` writes `build_report.json` and refuses a production pass when required checks fail:

- root USD composition, assets, meters, Z-up, ParticleField schema, and physics schemas;
- COLMAP/splat-to-proxy registration and RGB-D depth-to-collider error;
- manifold/UV object geometry, collider complexity, material files, and physical properties;
- hidden-background coverage/clean-plate evidence;
- held-out PSNR, SSIM, and optional LPIPS reports;
- Isaac headless static raycast, drop/rest, push, and kinematic pick/carry/place smoke tests.

---

## Legacy synthetic-YOLO pipeline overview

```text
video / frames
    │
    ▼ preprocess (optional) ──► workspace/frames/
    │
    ▼ reconstruct ──► ns-process-data (COLMAP) ──► workspace/ns_data/
    │                      └──► COLMAP TXT ──► workspace/colmap_txt/
    │                      └──► ns-train splatfacto ──► workspace/ns_outputs/.../config.yml
    │
    ▼ label ──► workspace/labels_real/          (YOLO pseudo-labels on real frames)
    ▼ lift  ──► workspace/objects_3d.npz        (3D AABBs from 2D + COLMAP)
    │
    ▼ synthesize ──► camera_path.json + labels_synthetic/ + workspace/renders/
    │
    ▼ export-dataset ──► YOLO dataset tree + data.yaml
    ▼ benchmark ──► workspace/reports/report_{A,B,C}.json
```

**Typical full run** (from repo root, venv active):

```bash
scan2usd doctor configs/example_scene.yaml
scan2usd preprocess configs/example_scene.yaml          # optional if reconstruct can read video
scan2usd reconstruct configs/example_scene.yaml
scan2usd label configs/example_scene.yaml
scan2usd lift configs/example_scene.yaml
# set splat_config_path in YAML to printed config.yml
scan2usd synthesize configs/example_scene.yaml
scan2usd export-dataset configs/example_scene.yaml --mode mixed
scan2usd benchmark configs/example_scene.yaml --experiment all
```

---

## Installation

### Python environment

```bash
cd /path/to/Scan2USD
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e ".[dev]"
```

- **Core package:** `scan2usd` (Typer CLI, OpenCV, Ultralytics, PyYAML, etc.).
- **Dev extras:** `pytest`, `ruff` (`pip install -e ".[dev]"`).
- **Nerfstudio (recommended):** install in the **same** venv, e.g. `pip install nerfstudio` or `pip install -e ".[nerfstudio]"` if you use the optional extra in `pyproject.toml`.

**Setuptools:** PyTorch often requires `setuptools<82`. If you see conflicts:

```bash
python -m pip install "setuptools>=61,<82"
```

### System dependencies

| Tool | Used by |
|------|---------|
| **COLMAP** | `ns-process-data`, COLMAP TXT export |
| **ffmpeg** / **ffprobe** | Nerfstudio data processing |
| **NVIDIA GPU + driver** | Splatfacto training (strongly recommended) |
| **CUDA toolkit (`nvcc`)** | gsplat JIT if no prebuilt wheel (see `doctor`) |

Verify everything:

```bash
scan2usd doctor configs/example_scene.yaml
```

On Debian/Ubuntu, `doctor` may print a single `sudo apt install …` line for missing packages.

### NumPy 2.x vs Nerfstudio

Ultralytics expects **NumPy 2.x**; `pip install nerfstudio` may pull NumPy 1.26. If YOLO/OpenCV break:

```bash
pip install "numpy>=2,<3"
pip install -e ".[dev]"
```

If `ns-train` then fails, use a **second venv** for Nerfstudio only and point `external.ns_*` in YAML to that venv’s scripts.

---

## Scene configuration (YAML)

Copy `configs/example_scene.yaml` per scene. All commands take:

```bash
scan2usd <command> path/to/scene.yaml
```

### Top-level fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | `default_scene` | Human-readable scene id (logging only). |
| `paths_relative_to` | string | *(cwd)* | How relative paths are resolved. See [Path resolution](#path-resolution). |
| `video_path` | path or `null` | `null` | Source video (`.mp4`, `.mov`, …). Used by `preprocess` and auto-extract in `reconstruct`. |
| `frames_dir` | path | `workspace/frames` | Extracted / input images for COLMAP and labeling. |
| `workspace_dir` | path | `workspace` | Root for labels, reports, camera paths, etc. |
| `colmap_txt_dir` | path | `workspace/colmap_txt` | Exported COLMAP text models (`cameras.txt`, `images.txt`, `points3D.txt`). |
| `nerfstudio_data_dir` | path | `workspace/ns_data` | Output of `ns-process-data` (`transforms.json`, `colmap/sparse/0`, images). |
| `splat_config_path` | path or `null` | `null` | Path to Nerfstudio `config.yml` after training. Required for `synthesize` renders and optional for `view`. |
| `renders_dir` | path | `workspace/renders` | Synthetic RGB from `ns-render` (`synthesize`). |
| `dataset_dir` | path | `workspace/dataset` | Reserved / mixed dataset root naming. |
| `classes` | list of strings | 7 indoor COCO-like names | YOLO class names (order = class id 0…N−1). |
| `yolo_model` | string | `yolov8n.pt` | Ultralytics weights for `label` and `benchmark`. |
| `train_epochs` | int | `50` | YOLO training epochs in `benchmark`. |
| `train_imgsz` | int | `640` | YOLO image size. |
| `train_batch` | int | `8` | YOLO batch size. |
| `seed` | int | `42` | RNG seed for pose sampling and dataset splits. |

### `split` — train/val for real YOLO dataset

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strategy` | `session` \| `random_frame` | `session` | How to split real frames. |
| `val_sessions` | list of strings | `[]` | If non-empty, these session names go to **val**; all others to train. |
| `val_ratio` | float | `0.2` | Fraction of **sessions** (or frames) in val when `val_sessions` is empty. |

**Session assignment** (from `records_from_frame_dir`):

- Image in `frames/session_a/img.jpg` → session `session_a`.
- Image directly in `frames/` → session `default` (single bucket).
- Filename `morning__0001.jpg` does **not** auto-parse `morning`; use **subfolders** per session or `random_frame` strategy.

### `pose_sampling` — novel cameras for `synthesize`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_poses` | int | `200` | Number of synthetic camera poses. |
| `position_jitter_m` | float | `0.05` | Random translation jitter (meters). |
| `height_jitter_m` | float | `0.02` | Vertical jitter (meters). |
| `max_rotation_deg` | float | `8.0` | Max rotation perturbation (degrees). |
| `interpolation_keyframes` | int | `8` | Smoothing between pose keyframes. |

### `lift` — 3D box merging

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `min_points_in_box` | int | `8` | Minimum COLMAP points inside a frustum AABB to keep a detection. |
| `merge_center_dist_m` | float | `0.35` | Merge 3D boxes whose centers are within this distance (meters). |

### `external` — binary overrides

| Key | Default | Description |
|-----|---------|-------------|
| `colmap` | `colmap` | COLMAP executable or full path. |
| `ns_process_data` | `ns-process-data` | Nerfstudio process-data CLI. |
| `ns_train` | `ns-train` | Nerfstudio train CLI. |
| `ns_render` | `ns-render` | Nerfstudio render CLI. |

Resolution order: **explicit path in YAML** → `which` on `PATH` → `python -m nerfstudio.scripts.*` if the package is importable.

`scan2usd view` also resolves `ns-viewer` (not listed in default YAML; add `ns_viewer: ns-viewer` if you need a custom path).

### `process_data` — passed to `ns-process-data images`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_downscales` | int | `3` | Image pyramid depth (÷2 each level). **Lower = sharper** inputs (e.g. `2` or `1`). More VRAM/disk. |

### `splatfacto` — passed to `ns-train splatfacto`

These map to Nerfstudio CLI flags (see [Nerfstudio docs](https://docs.nerf.studio/) for full semantics).

| Key | Type | Default | Nerfstudio flag | Description |
|-----|------|---------|-----------------|-------------|
| `max_num_iterations` | int | `30000` | `--max-num-iterations` | Training steps. |
| `experiment_name` | string | `splatfacto` | `--experiment-name` | Output subfolder under `ns_outputs/`. |
| `steps_per_log` | int | `50` | `--logging.steps-per-log` | Terminal log interval (with `max-log-size 0`). |
| `use_bilateral_grid` | bool | `false` | `--pipeline.model.use-bilateral-grid` | Per-view exposure/color correction (helps uneven lighting). |
| `background_color` | `random` \| `black` \| `white` | `random` | `--pipeline.model.background-color` | Training background. Use `black` for dark rooms. |
| `rasterize_mode` | `classic` \| `antialiased` | `classic` | `--pipeline.model.rasterize-mode` | `antialiased` reduces aliasing at different scales. |
| `cull_alpha_thresh` | float | `0.1` | `--pipeline.model.cull-alpha-thresh` | Lower (e.g. `0.005`) keeps faint Gaussians in shadows. |
| `ssim_lambda` | float | `0.2` | `--pipeline.model.ssim-lambda` | SSIM vs L1 mix (e.g. `0.15` for more L1). |
| `camera_res_scale_factor` | float or omit | *(omit)* | `--pipeline.datamanager.camera-res-scale-factor` | Scale loaded images (`1.0` full; `0.5` if GPU OOM). |
| `model_num_downscales` | int or omit | *(omit)* | `--pipeline.model.num-downscales` | Multi-res **training** schedule (default in Nerfstudio: `2`). |
| `camera_optimizer_mode` | string | `off` | `--pipeline.model.camera-optimizer.mode` | `off`, `SO3xR3`, or `SE3` for mild pose refinement. |

**Note:** Input resolution is controlled by `process_data.num_downscales` and Nerfstudio’s auto downscale (logged as “Auto image downscale factor of …”). There is **no** `dataparser_downscale_factor` in the `ns-train splatfacto` CLI.

**Example tuning block** (uneven lighting / shadows):

```yaml
process_data:
  num_downscales: 2
splatfacto:
  max_num_iterations: 40000
  use_bilateral_grid: true
  background_color: black
  rasterize_mode: antialiased
  cull_alpha_thresh: 0.005
  ssim_lambda: 0.15
  camera_optimizer_mode: "off"
```

---

## Path resolution

| `paths_relative_to` | Base for relative paths |
|---------------------|-------------------------|
| *(unset)* | **Current working directory** (where you run `scan2usd`) |
| `config` | Directory containing the YAML file |
| `/absolute/path` or `~/path` | Explicit base |

**Recommendation:** run commands from the repo root so `workspace/` in YAML means `<repo>/workspace/`.

Path keys resolved this way: `video_path`, `frames_dir`, `workspace_dir`, `colmap_txt_dir`, `nerfstudio_data_dir`, `splat_config_path`, `renders_dir`, `dataset_dir`.

---

## Workspace layout

After a full pipeline (default paths):

```text
workspace/
  frames/                    # input images (frame_000000.jpg, …)
  colmap_txt/                # cameras.txt, images.txt, points3D.txt
  ns_data/                   # Nerfstudio dataset
    images/
    transforms.json
    colmap/sparse/0/         # binary COLMAP model
  ns_outputs/
    splatfacto/splatfacto/<timestamp>/
      config.yml             # set splat_config_path to this
      nerfstudio_models/     # checkpoints
  labels_real/               # YOLO .txt per frame (from label)
  objects_3d.npz             # lifted 3D boxes (from lift)
  labels_synthetic/          # YOLO labels for synthetic views
  renders/                   # synthetic PNG/JPG (from synthesize)
  camera_path.json           # novel poses for ns-render
  camera_path_sanity_real.json  # optional (sanity-cam)
  dataset_real/              # YOLO layout for experiment A
  dataset_synthetic/         # synthetic train + real val (B)
  dataset/ or mixed root     # experiment C
  reports/
    report_A.json
    report_B.json
    report_C.json
    summary_ABC.json
```

---

## Commands reference

Global: `scan2usd --help`, `scan2usd <command> --help`.

Every command requires: **`CONFIG.yaml`** (first positional argument).

---

### `scan2usd clean`

**Purpose:** Remove experiment artifacts for a scene (datasets, reports, labels, renders, or the whole workspace).

```bash
scan2usd clean configs/example_scene.yaml --tier light
scan2usd clean configs/example_scene.yaml --tier medium -y
scan2usd clean configs/example_scene.yaml --tier full --dry-run
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--tier` | string | **required** | `light` \| `medium` \| `full` (see below). |
| `--dry-run` | flag | off | Print paths that would be removed. |
| `--yes` / `-y` | flag | off | Skip confirmation for `--tier full`. |
| `--ultralytics-runs` / `--no-ultralytics-runs` | flag | on | Also delete `./runs/` under the current working directory. |
| `--downloads` | flag | off | Also delete `yolo26n.pt` / `yolov8n.pt` in cwd if present. |

| Tier | Removes |
|------|---------|
| **light** | `dataset_real`, `dataset_synthetic`, `dataset_mixed`, `reports`, optional `dataset_dir` if outside workspace |
| **medium** | light + `labels_real`, `labels_synthetic`, `objects_3d.npz`, `camera_path*.json`, `renders_dir` |
| **full** | Entire `workspace_dir` (+ `frames_dir`, `colmap_txt_dir`, `nerfstudio_data_dir`, `renders_dir`, `dataset_dir` if configured outside workspace) |

After **full**, clear or update `splat_config_path` in YAML before the next `reconstruct`.

Makefile shortcuts: `make clean-light`, `make clean-medium`, `make clean-full` (pass `-y`).

---

### `scan2usd doctor`

**Purpose:** Check COLMAP, ffmpeg, Nerfstudio CLIs, GPU, Python imports (torch, gsplat, ultralytics, nerfstudio). Print apt hints on Linux.

```bash
scan2usd doctor configs/example_scene.yaml
```

| Argument | Required | Description |
|----------|----------|-------------|
| `CONFIG` | yes | Scene YAML (used for `external.*` resolution). |

**No options.**

---

### `scan2usd preprocess`

**Purpose:** Extract frames from `video_path` with blur filtering and optional subsampling.

```bash
scan2usd preprocess configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stride` | int | `1` | Keep every Nth frame from the video (before blur filter). |
| `--keyframe-every` | int | `1` | After extraction, keep every Nth **kept** frame. |
| `--max-frames` | int or omit | none | Stop after this many saved frames. |

**Requires:** `video_path` exists.

**Output:** `frames_dir/*.jpg` (`frame_000000.jpg`, …).

Blur rejection: Laplacian variance below **50** (fixed in code) drops blurry frames.

Supported video extensions (typical): `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, … (OpenCV + FFmpeg).

---

### `scan2usd reconstruct`

**Purpose:** Run `ns-process-data images` → export COLMAP TXT → optional `ns-train splatfacto`.

```bash
scan2usd reconstruct configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--skip-train` | flag | off | Only COLMAP / process-data + TXT export. |
| `--skip-process-data` | flag | off | Reuse existing `nerfstudio_data_dir/colmap/sparse/0`; skip `ns-process-data`. |
| `--max-iterations` | int or omit | YAML `splatfacto.max_num_iterations` | Override training length. |
| `--viewer` | flag | off | Enable Nerfstudio live Web viewer during training (noisier logs). |
| `--video-stride` | int | `15` | When auto-extracting from `video_path`: keep every Nth sharp frame. `1` = all (slow COLMAP); `15`–`30` is typical for ~30 fps video. |
| `--video-max-frames` | int | `600` | Cap auto-extracted frames (`0` = no cap). Avoids 10k+ near-duplicate frames that often yield almost no COLMAP registrations. |

**Frame input:**

1. If `frames_dir` has images → use them.
2. Else if `video_path` exists → extract with `--video-stride` and `--video-max-frames` into `frames_dir` (not every frame by default).
3. Else → error (run `preprocess` first).

For full control (blur filter + `keyframe_every`), use `scan2usd preprocess` before `reconstruct`.

**Subprocess behavior:**

- Prepends `/usr/local/cuda/bin` (and variants) for gsplat JIT.
- Sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for PyTorch 2.6+ checkpoint loading.
- Training uses `--vis tensorboard` unless `--viewer`.
- Quiet logging: `--logging.local-writer.max-log-size 0`.

**Outputs:**

- `colmap_txt_dir/`
- `nerfstudio_data_dir/`
- `workspace/ns_outputs/<experiment_name>/splatfacto/<timestamp>/config.yml` (if training)

**Common recipes:**

```bash
# Full pipeline
scan2usd reconstruct configs/example_scene.yaml

# COLMAP only
scan2usd reconstruct configs/example_scene.yaml --skip-train

# Re-train with new splatfacto: YAML settings (keep COLMAP)
scan2usd reconstruct configs/example_scene.yaml --skip-process-data
```

---

### `scan2usd view`

**Purpose:** Open trained Splatfacto in Nerfstudio Web viewer (`ns-viewer`). By default, overlays wireframe **3D bounding boxes** from `workspace/objects_3d.npz` (same world frame as the splat; run `scan2usd lift` first).

```bash
scan2usd view configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--load-config` | path | see below | `config.yml` from a training run. |
| `--boxes` / `--no-boxes` | flag | `--boxes` | Draw lifted AABBs in the viewer (viser wireframes). |
| `--objects-npz` | path | `workspace/objects_3d.npz` | NPZ from `scan2usd lift`. |

**Config resolution:** `--load-config` → else `splat_config_path` in YAML → else newest `workspace/ns_outputs/<experiment_name>/splatfacto/*/config.yml`.

Open the printed URL (usually **http://localhost:7007**). Use the **Show 3D boxes** checkbox in the sidebar to toggle box visibility.

3D boxes are overlaid in the **trained splat coordinate frame**, then scaled to match Nerfstudio's viser scene (×10, same as camera frustums). The NPZ stores ``transforms.json`` / ``sparse_pc.ply`` coordinates for synthesis. Re-run ``scan2usd view`` after upgrading (``lift`` only if boxes themselves changed).

---

### `scan2usd label`

**Purpose:** Run Ultralytics detector on real frames; write YOLO labels mapped to `classes` in YAML.

```bash
scan2usd label configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--weights` | string | `yolov8n.pt` | Ultralytics model path or name. |

**Output:** `workspace/labels_real/<label_key>.txt` (one file per frame; normalized YOLO format).

**Requires:** images in `frames_dir`; COCO-pretrained model can only detect COCO-like categories (mapped to your class list).

---

### `scan2usd lift`

**Purpose:** Fuse 2D YOLO boxes with sparse SfM points into merged 3D axis-aligned bounding boxes.

```bash
scan2usd lift configs/example_scene.yaml
```

**Requires:** `labels_real/` from `label`, and `nerfstudio_data_dir/` with `transforms.json` + `sparse_pc.ply` from `reconstruct`.

**Output:** `workspace/objects_3d.npz` (`class_id`, `bbox_min`, `bbox_max`, `obb_center`, `obb_half`, `obb_rotation`, `coord_frame=nerfstudio`). Boxes are **oriented** (PCA on SfM points per detection), not world-axis cubes. The viewer maps them into the splat frame and draws wireframes at Nerfstudio's viser scale (×10).

**Config:** `lift.min_points_in_box`, `lift.merge_center_dist_m`, `lift.depth_trim_mad`, `lift.percentile`, optional `lift.max_extent_m`.

#### How lift works (coordinate frames)

```text
2D YOLO label on frame
    → find sparse SfM points from sparse_pc.ply projecting inside the 2D rectangle
    → depth-trim points along the viewing ray (drop background outliers)
    → fit oriented 3D box (PCA or view-aligned: camera + scene “up” + 2D box aspect)
    → merge nearby same-class boxes across frames
    → save objects_3d.npz in transforms.json / sparse_pc.ply space

scan2usd view
    → load splat checkpoint + dataparser (orient / center / scale)
    → map boxes to viewer coords (orient-only, not double applied_transform)
    → multiply by Nerfstudio viser scale (×10) and draw wireframes
```

**Important:** `view` can look wrong even when lift is OK. Always debug lift on **real 2D images** first.

#### `scan2usd debug-lift`

Writes per-detection **side-by-side** images under `workspace/lift_debug/`:

| Panel | Meaning |
|-------|---------|
| **Left** | YOLO pseudo-label only (reference — usually better) |
| **Right** | 3D lift for **that** detection only (not merged) |
| **Cyan** (right) | Sparse SfM points inside the 2D box |
| **Red rect** (right) | 3D AABB reprojected (IoU metric) |
| **Blue hull** (right) | Oriented 3D box reprojected |

```bash
scan2usd debug-lift configs/example_scene.yaml
# optional: --max-frames 24 --frame-stride 15 --out-dir workspace/lift_debug
```

Read `workspace/lift_debug/summary.json` — field `iou_reproj` (median should be ~0.3+ if lift is sane for that frame).

| Symptom | Likely cause |
|---------|----------------|
| Right panel nowhere near left (YOLO) | Lift / sparse points (walls along ray, huge `n_sfm_inliers`) |
| Right hugs left in debug, wrong in `view` | Dataparser or viser ×10 transform |
| YOLO good, lift bad (your case) | Expected with sparse SfM; tune `inner_margin_frac`, `depth_trim_mad`, lower `merge_center_dist_m` |

---

### `scan2usd sanity-cam`

**Purpose:** Export a short camera path from **real** training poses for manual `ns-render` checks.

```bash
scan2usd sanity-cam configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--count` | int | `5` | Number of poses from start of trajectory. |

**Output:** `workspace/camera_path_sanity_real.json`.

**Example render:**

```bash
ns-render camera-path \
  --load-config workspace/ns_outputs/splatfacto/splatfacto/<run>/config.yml \
  --camera-path-filename workspace/camera_path_sanity_real.json \
  --output-path workspace/renders/sanity \
  --output-format images
```

---

### `scan2usd synthesize`

**Purpose:** Sample novel poses → write `camera_path.json` → optional synthetic YOLO labels → optional `ns-render`.

```bash
scan2usd synthesize configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--skip-render` | flag | off | Only camera path + labels; no `ns-render`. |

**Requires:**

- `transforms.json` under `nerfstudio_data_dir` (from `reconstruct`).
- For labels: `objects_3d.npz` (`lift` first).
- For renders: valid `splat_config_path` unless `--skip-render`.

**Outputs:**

- `workspace/camera_path.json`
- `workspace/labels_synthetic/synth_XXXXXX.txt`
- `workspace/renders/*` (if rendering)

---

### `scan2usd export-dataset`

**Purpose:** Build Ultralytics YOLO directory tree + `data.yaml`.

```bash
scan2usd export-dataset configs/example_scene.yaml --mode MODE
```

| Option | Type | Required | Values | Description |
|--------|------|----------|--------|-------------|
| `--mode` | string | **yes** | `real` \| `synthetic` \| `mixed` | Which dataset layout to emit. |

| Mode | Train source | Val source | Typical experiment |
|------|--------------|------------|-------------------|
| `real` | Real frames + `labels_real` | Held-out real (split) | **A** |
| `synthetic` | Synthetic renders + `labels_synthetic` | Real val (copied from real dataset) | **B** |
| `mixed` | Real train + synthetic train | Real val | **C** |

**Outputs:** prints path to `data.yaml` on stdout.

Roots (defaults):

- `workspace/dataset_real/`
- `workspace/dataset_synthetic/`
- mixed under `workspace/dataset/` (via `build_mixed_dataset`)

---

### `scan2usd benchmark`

**Purpose:** Train and validate YOLO per experiment; write JSON reports.

```bash
scan2usd benchmark configs/example_scene.yaml [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--experiment` | string | `all` | `A`, `B`, `C`, or `all` (case-insensitive). |

**Uses YAML:** `yolo_model`, `train_epochs`, `train_imgsz`, `train_batch`, `seed`.

**Outputs:**

- `workspace/reports/report_A.json` (etc.)
- `workspace/reports/summary_ABC.json` when `--experiment all` (includes `goal_C_gt_A`: mAP(C) > mAP(A))

**Prerequisites:**

- **A:** `export-dataset --mode real` (or benchmark builds it).
- **B:** synthetic renders + labels + real val.
- **C:** mixed dataset materialized.

---

## Makefile shortcuts

From repo root (assumes venv on `PATH`):

| Target | Command |
|--------|---------|
| `make install` | `pip install -e ".[dev]"` |
| `make doctor` | `scan2usd doctor configs/example_scene.yaml` |
| `make preprocess` | `scan2usd preprocess configs/example_scene.yaml` |
| `make reconstruct` | `scan2usd reconstruct configs/example_scene.yaml` |
| `make label` | `scan2usd label configs/example_scene.yaml` |
| `make lift` | `scan2usd lift configs/example_scene.yaml` |
| `make synthesize` | `scan2usd synthesize configs/example_scene.yaml` |
| `make clean-light` | `scan2usd clean … --tier light -y` |
| `make clean-medium` | `scan2usd clean … --tier medium -y` |
| `make clean-full` | `scan2usd clean … --tier full -y` |
| `make export` | `export-dataset --mode mixed` |
| `make benchmark` | `benchmark --experiment all` |
| `make test` | unit tests (`pytest -m "not e2e"`) |
| `make test-e2e` | end-to-end smoke test |

---

## Experiments A / B / C

| Exp | Train on | Validate on | Question |
|-----|----------|-------------|----------|
| **A** | Real only | Real val | Baseline detector on real data. |
| **B** | Synthetic only | **Same** real val | Does synthetic training generalize? |
| **C** | Real + synthetic train | Real val | Does mixing beat A? |

**Goal metric:** `summary_ABC.json` → `goal_C_gt_A` is true when mAP(C) > mAP(A).

---

## Nerfstudio tools (direct)

Scan2USD wraps these; you can also run them manually with the same env:

| Tool | Role |
|------|------|
| `ns-process-data images` | COLMAP + `transforms.json` |
| `ns-train splatfacto` | Gaussian splat training |
| `ns-render camera-path` | Render along JSON path |
| `ns-viewer` | Interactive viewer |

**PyTorch 2.6+ checkpoint loading:**

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

(`scan2usd` subprocesses set this automatically.)

**Interpolated trajectory (no Scan2USD wrapper):**

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ns-render interpolate \
  --load-config workspace/ns_outputs/.../config.yml \
  --output-path workspace/renders/interpolate \
  --output-format images
```

---

## Environment variables

| Variable | Set by Scan2USD | Purpose |
|----------|-----------------|--------|
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | yes (subprocesses) | Allow Nerfstudio `.ckpt` load on PyTorch 2.6+. |
| `VIRTUAL_ENV` | user | Used to find venv `python` for ninja shim. |
| `XDG_CACHE_HOME` | user | Ninja shim under `$XDG_CACHE_HOME/scan2usd/`. |
| `CUDA_HOME` / `CUDA_PATH` | user | Doctor / gsplat: locate `nvcc`. |
| `PATH` | augmented | Prepends `/usr/local/cuda/bin` when `nvcc` exists there. |

---

## Troubleshooting

| Symptom | What to try |
|---------|-------------|
| `gsplat CUDA` missing in doctor | Install CUDA toolkit; ensure `nvcc` under `/usr/local/cuda/bin` or on `PATH`. |
| `externally-managed-environment` (pip) | Use venv; do not `pip install` system-wide. |
| `ns-train` unrecognized `--pipeline.datamanager.dataparser.*` | Remove invalid keys from YAML; use `process_data.num_downscales` and `camera_res_scale_factor`. |
| `torch.load` / `weights_only` error in `ns-viewer` | Use `scan2usd view` or `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. |
| Noisy training log spam | Default fixed (`max-log-size 0`); avoid `--viewer` unless needed. |
| `reconstruct` fails with "registered only N/M frames" | Working as intended — SfM failed and everything downstream would be fit to a few views. Extract denser/sharper (`--video-stride 8 --blur-threshold 100`) or recapture with slower motion and locked exposure/focus. `--min-registration-rate 0` overrides. |
| Many frames "dropped below blur threshold" | The video is soft. Lowering `--blur-threshold` keeps more frames but they still won't match in COLMAP; recapture instead. |
| COLMAP low % registered | More overlap, slower motion, more texture; fewer blurry frames (`preprocess` stride). |
| COLMAP registered ≈0% after long video | Too many near-duplicate frames: use `reconstruct --video-stride 15 --video-max-frames 600` or `preprocess --stride …` then `reconstruct`. |
| `CUDA error: invalid argument` early in `ns-train` | Fix COLMAP first (need many registered views). Reboot; `nvidia-smi`; try `CUDA_LAUNCH_BLOCKING=1`; update driver / PyTorch+CUDA stack for RTX 50. |
| Dark / inconsistent regions in splat | `use_bilateral_grid: true`, locked exposure on re-capture; see `splatfacto` table. |
| OOM during training | `camera_res_scale_factor: 0.5`, higher `process_data.num_downscales`, fewer frames. |
| `synthesize` skips render | Set `splat_config_path` to trained `config.yml`. |
| No 3D objects after `lift` | Run `label` first; check COLMAP TXT and `min_points_in_box`. |

---

## Capture guidance

See [capture_sop.md](capture_sop.md) for recording quality, overlap, exposure, and recommended frame naming / session folders.

---

## Getting help from the CLI

```bash
scan2usd --help
scan2usd reconstruct --help
# … per-command --help
```

For Nerfstudio-specific flags beyond what YAML exposes:

```bash
ns-train splatfacto --help
ns-process-data images --help
```
