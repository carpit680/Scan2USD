# Scan2USD (MVP)

Offline pipeline: **RGB capture → COLMAP/Nerfstudio (Splatfacto) → synthetic viewpoints → YOLO labels → A/B/C benchmark**.

**Full manual:** [docs/USAGE.md](docs/USAGE.md) — every command, YAML option, workspace layout, experiments, and troubleshooting.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -e ".[dev]"
```

PyTorch (pulled in by Ultralytics) currently pins **setuptools to versions below 82**. Avoid `pip install -U setuptools` with no upper bound, or pip will report a conflict with `torch`. This package declares `setuptools>=61,<82` so editable installs stay compatible.

If setuptools is already at 82+, run: `python -m pip install "setuptools>=61,<82"`.

### Paths in YAML

Relative paths (`workspace/`, etc.) resolve against the **current working directory** (the directory you run `scan2usd` from), not the folder that contains the YAML. Run commands from the repo root, or use absolute paths.

To resolve relative to the config file instead (old behavior), set `paths_relative_to: config` at the top of the YAML.

`reconstruct` will **extract frames from `video_path`** automatically if `frames_dir` is missing or has no images (so you can skip a separate `preprocess` step when a video is configured).

### External tools (COLMAP, Nerfstudio, ffmpeg)

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

### Python deps: Ultralytics + Nerfstudio in one venv

Ultralytics installs **`opencv-python`**, which expects **NumPy 2.x**. A plain **`pip install nerfstudio`** can downgrade NumPy to **1.26** (transitive deps such as **nuscenes-devkit**), which triggers pip’s warning: *opencv-python requires numpy>=2, but you have numpy 1.26.4*.

**If you care most about YOLO / OpenCV**, re-upgrade NumPy after Nerfstudio:

```bash
pip install "numpy>=2,<3"
pip install -e ".[dev]"   # re-apply scan2usd pins if needed
```

Pip may then warn that **nuscenes-devkit** wants NumPy below 2; many Nerfstudio workflows still run. If `ns-train` or imports fail, use a **second environment** for `ns-*` only and set `external.ns_process_data`, `external.ns_train`, and `external.ns_render` to that env’s scripts.

1. Put frames under `workspace/frames/` (or set `video_path` to an **MP4, MOV**, or other FFmpeg-backed file and run `scan2usd preprocess`).
2. Run **`scan2usd doctor`** and install anything reported missing.
3. Run stages:

```bash
scan2usd reconstruct configs/example_scene.yaml
scan2usd label configs/example_scene.yaml
scan2usd lift configs/example_scene.yaml
```

4. Train splats (inside Nerfstudio env), then set `splat_config_path` in the YAML to the exported `config.yml` from `ns-train`.

```bash
scan2usd synthesize configs/example_scene.yaml
scan2usd export-dataset configs/example_scene.yaml --mode mixed
scan2usd benchmark configs/example_scene.yaml --experiment all
```

Cleanup between runs: `scan2usd clean configs/example_scene.yaml --tier light|medium|full` (see [docs/USAGE.md](docs/USAGE.md#scan2usd-clean)).

See [docs/capture_sop.md](docs/capture_sop.md) for capture guidance.

### Tests

- **`make test`** — unit tests only (`python -m pytest -m "not e2e"`).
- **`make test-e2e`** — full **lift → synthesize → export → benchmark (B)** on a synthetic in-repo workspace (first run may download YOLO weights).

### More documentation

- **[docs/USAGE.md](docs/USAGE.md)** — complete command reference, all YAML keys, `process_data` / `splatfacto` tuning, Makefile, experiments A/B/C, troubleshooting.
- **[docs/capture_sop.md](docs/capture_sop.md)** — how to record video for COLMAP / splats.
