# Scan2USD

Hybrid scan compiler for **Isaac Sim 6.x**:

```text
room/workcell capture
  → metric camera/scene registration
  → photoreal 3DGRUT Gaussian ParticleField (visual only)
  → dense static triangle collision geometry
  → reviewed, separately reconstructed rigid-object meshes
  → baked + relightable PBR material variants
  → layered OpenUSD scene with PhysX rigid bodies and validation reports
```

The older Splatfacto → synthetic-YOLO benchmark commands remain available as a legacy workflow. They are not used to generate production USD assets.

**Full manual:** [docs/USAGE.md](docs/USAGE.md).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e ".[dev,geometry,review]"
```

PyTorch (pulled in by Ultralytics) currently pins **setuptools to versions below 82**. Avoid `pip install -U setuptools` with no upper bound, or pip will report a conflict with `torch`. This package declares `setuptools>=61,<82` so editable installs stay compatible.

If setuptools is already at 82+, run: `python -m pip install "setuptools>=61,<82"`.

### Paths in YAML

Relative paths (`workspace/`, etc.) resolve against the **current working directory** (the directory you run `scan2usd` from), not the folder that contains the YAML. Run commands from the repo root, or use absolute paths.

To resolve relative to the config file instead (old behavior), set `paths_relative_to: config` at the top of the YAML.

`reconstruct` will **extract frames from `video_path`** automatically if `frames_dir` is missing or has no images, using a **subsampled** default (`--video-stride 15`, `--video-max-frames 600`) so COLMAP is not flooded with near-duplicate frames. Use `scan2usd preprocess` for full control, or pass e.g. `--video-stride 1 --video-max-frames 0` only if you know you need dense sampling.

### External tools

`reconstruct` / `synthesize` need **COLMAP** and **Nerfstudio** CLIs. **ffmpeg** must also be on `PATH`; Nerfstudio's `ns-process-data` calls it even for the `images` flow (resize / transcode paths).

Install COLMAP from your OS or [colmap.github.io](https://colmap.github.io/). Install Nerfstudio in the **same** venv you use for `scan2usd` (so `import nerfstudio` works), e.g. `pip install nerfstudio`, or put the `ns-*` scripts on `PATH`. Install ffmpeg from your OS or [ffmpeg.org](https://ffmpeg.org/download.html) (e.g. `sudo apt install ffmpeg` on Ubuntu).

If the `ns-process-data` script is not on `PATH` but the package is installed, Scan2USD will fall back to `python -m nerfstudio.scripts.process_data` (and the same pattern for `train` / `render`).

Check what your machine resolves to:

```bash
scan2usd doctor configs/example_scene.yaml
```

On **Linux**, doctor prints a single **`sudo apt install …`** line aggregating missing Debian/Ubuntu packages (e.g. `colmap`, `ffmpeg`, `git`, OpenCV runtime libs, **`nvidia-cuda-toolkit`** when `nvcc` is needed for **gsplat** / Splatfacto). It also checks **ffprobe**, **curl** / **wget**, **nvidia-smi** (optional), and Python imports (**numpy**, **cv2**, **torch**, **`torch.cuda`**, **gsplat** + `nvcc` / prebuilt `csrc`, **ultralytics**, **nerfstudio**).

**Splatfacto:** many `gsplat` wheels do not include a prebuilt CUDA module; gsplat then needs **`nvcc`** (CUDA toolkit) to JIT-compile. If `doctor` reports **gsplat CUDA** missing, install a toolkit whose CUDA major matches your PyTorch wheel (e.g. cu12) — `sudo apt install nvidia-cuda-toolkit` is a common starting point on Ubuntu, or use [NVIDIA’s CUDA installer](https://developer.nvidia.com/cuda-downloads).

You can override binaries in YAML under `external:` (`colmap`, `ns_process_data`, `ns_train`, `ns_render`) with full paths.

Production USD builds deliberately isolate heavyweight/incompatible runtimes. Configure:

- NVIDIA **3DGRUT v1.1+** for standard `UsdVol.ParticleField3DGaussianSplat`.
- **nvblox** for RGB-D/LiDAR static geometry, or **OpenMVS** for RGB-only dense geometry.
- A **SAM2** mask-propagation runner and masked object-reconstruction runner.
- Isaac Sim 6.x `python.sh` for USDC conversion and headless physics validation.

`scan2usd doctor` reports each configured production tool separately.

### Python deps: Ultralytics + Nerfstudio in one venv

Ultralytics installs **`opencv-python`**, which expects **NumPy 2.x**. A plain **`pip install nerfstudio`** can downgrade NumPy to **1.26** (transitive deps such as **nuscenes-devkit**), which triggers pip’s warning: *opencv-python requires numpy>=2, but you have numpy 1.26.4*.

**If you care most about YOLO / OpenCV**, re-upgrade NumPy after Nerfstudio:

```bash
pip install "numpy>=2,<3"
pip install -e ".[dev]"   # re-apply scan2usd pins if needed
```

Pip may then warn that **nuscenes-devkit** wants NumPy below 2; many Nerfstudio workflows still run. Keep 3DGRUT, SAM2, nvblox, and Isaac in their own environments and configure their command paths instead of merging all dependencies into this venv.

### Hybrid USD workflow

1. Configure the `capture`, `reconstruction`, `segmentation`, `geometry`, `materials`, `physics`, `usd`, `qa`, and `external` sections in `configs/example_scene.yaml`.
2. Reconstruct camera poses, then initialize and segment the USD build:

```bash
scan2usd reconstruct configs/example_scene.yaml
scan2usd init-usd configs/example_scene.yaml --mode production
scan2usd segment-usd configs/example_scene.yaml
scan2usd review configs/example_scene.yaml
```

3. RGB-only captures require an approved COLMAP→USD similarity transform (Z-up, meters):

```bash
scan2usd apply-metric-scale configs/example_scene.yaml \
  --known-length-m 0.91 --source-length 2.45 --reviewer NAME
# Or: scan2usd set-metric-transform configs/example_scene.yaml calibration.json --reviewer NAME
```

4. Build/resume. The command intentionally pauses when mask, physical-property, material, or lighting approval is required:

```bash
scan2usd build-usd configs/example_scene.yaml
scan2usd review configs/example_scene.yaml
scan2usd build-usd configs/example_scene.yaml
scan2usd validate-usd configs/example_scene.yaml --held-out-renders /path/to/isaac/renders
```

The root stage is `workspace/usd/scene.usd`; `workspace/usd/build_report.json` states whether it passed the production gates.

See [docs/capture_sop.md](docs/capture_sop.md) for capture guidance.

### GUI

A separate FastAPI + React UI lives under [`gui/`](gui/). It exposes every CLI command, YAML parameter (with tooltips), pipeline stages, review gates, doctor, and an in-app guide. Gradio `scan2usd review` is unchanged.

```bash
pip install -e gui/backend
cd gui/frontend && npm install && cd ../..
make gui   # API :8765 + Vite :5173
```

See [gui/README.md](gui/README.md).

### Tests

- **`make test`** — unit tests only (`python -m pytest -m "not e2e"`).
- **`make test-e2e`** — legacy synthetic-YOLO smoke test.

### More documentation

- **[docs/USAGE.md](docs/USAGE.md)** — hybrid USD workflow, command reference, validation, and legacy commands.
- **[docs/capture_sop.md](docs/capture_sop.md)** — how to record video for COLMAP / splats.
