"""Guide content served to the frontend."""

from __future__ import annotations

from typing import Any

GUIDE_SECTIONS: list[dict[str, Any]] = [
    {
        "id": "getting-started",
        "title": "Getting started",
        "body": """## Getting started (first time)

Scan2USD turns a room video into a layered **OpenUSD** scene for Isaac Sim (visuals + collision + reviewed objects).

1. On **Project**, click **Start from example** (or open your own YAML).
2. Open **Config → Essentials**. Set **Source video** with Browse (or **From phone…** QR upload on the same Wi‑Fi), check **Object types**, and **Save YAML**.
3. Open **Doctor** and fix anything marked missing (COLMAP, ffmpeg, GPU tools, …).
4. Go to **Pipeline** and run steps in order, starting with **Build cameras from your video**.
5. When the build asks for review, use the **Review** page to approve masks.

You do **not** need to edit YAML by hand. The GUI saves a normal scene config file Scan2USD already understands.
""",
    },
    {
        "id": "sample-workflow",
        "title": "Sample workflow",
        "body": """## Sample workflow (hybrid USD)

Use this checklist for a typical RGB walk-around of a room:

1. **Project** — open or create a scene YAML; set working directory to the repo root.
2. **Config → Essentials** — pick your `.mp4`, set scene name, keep modality `rgb`, add a clean-plate folder if you have one → **Save**.
3. **Doctor** — confirm core tools look ready.
4. **Pipeline → Build cameras from your video** (`reconstruct`). Wait until it finishes.
5. **Start a USD project** (`init-usd`) then **Find objects & masks** (`segment-usd`).
6. **Review** — set keepers to `approved` in the status dropdown (reject the rest).
7. **Stand the room upright** (`align-floor`). For RGB-only, **Apply metric scale** (`apply-metric-scale`) with a measured length (or meters-per-unit). Prefer that over raw `set-metric-transform`.
8. **Build the Isaac scene** (`build-usd`). If it exits with code **2**, return to Review and mark keepers approved, then run Build again.
9. **Validate quality** (`validate-usd`).

Deep links: [Config](/config) · [Pipeline](/pipeline) · [Review](/review) · [Doctor](/doctor)
""",
    },
    {
        "id": "setup",
        "title": "Setup & doctor",
        "body": """## Setup

1. Create a venv and install Scan2USD: `pip install -e ".[dev,geometry,review]"`.
2. Install this GUI: `pip install -e gui/backend` and `cd gui/frontend && npm install`.
3. Run `make gui` and open http://127.0.0.1:5173.

**Doctor** checks COLMAP, Nerfstudio CLIs, ffmpeg, optional 3DGRUT / SAM2 / Isaac paths, and Python packages. Use **Config → External tools** only if a binary is installed somewhere unusual.
""",
    },
    {
        "id": "paths",
        "title": "Paths & working directory",
        "body": """## Paths (beginner version)

When you open a project you choose:

- **Scene YAML** — the settings file.
- **Working directory (cwd)** — the folder relative paths like `workspace_desk/` are resolved from.

Set **Workspace folder** once (for example `workspace_desk`). Frames, `ns_data`, masks, USD output, uploads, and `build/` all live under that folder automatically — you do not need to set each path separately.

Default: leave **Where relative paths start from** as `cwd`, and set cwd to your Scan2USD repo root. Then `workspace_desk/` means `<repo>/workspace_desk`.

Only switch to `config` if you intentionally store paths relative to the YAML's folder. Advanced overrides for individual folders (frames, masks, …) are optional and live under Config → Advanced → Paths.

### Upload video from phone (LAN)

1. On the desktop run `make gui-lan` (API listens on `0.0.0.0:8765`).
2. Open a project, then **Config → Source video → Browse → From phone…**.
3. Scan the QR with your phone (same Wi‑Fi). Allow firewall port **8765** if needed.
4. Pick a video on the phone; the desktop dialog fills `video_path` when the upload finishes.
""",
    },
    {
        "id": "faq",
        "title": "FAQ",
        "body": """## FAQ

**Do I need to know YAML?**  
No. Use Config with sliders and Browse, then Save. YAML is written for you.

**RGB vs RGB-D vs LiDAR?**  
RGB is a normal camera video (most users). RGB-D adds depth maps. LiDAR uses a laser scan. RGB-only needs a metric scale / transform step before production.

**Why did Build stop with exit code 2?**  
Production mode requires human approval. Open **Review**, approve segmentation/assets/lighting, then run Build again.

**What does Doctor check?**  
External programs (COLMAP, ffmpeg, ns-*), hybrid tools (3DGRUT, SAM2, Isaac), and Python imports (torch, ultralytics, …).

**Clean light / medium / full?**  
Deletes regenerable workspace junk. Use **dry run** first. Full is most aggressive.

**Gradio vs this GUI for review?**  
Both use the same ReviewSession backend. Prefer this GUI's **Review** page. Gradio remains available via Commands → review.

**Do I need a GPU / Isaac?**  
Training and 3DGRUT need a CUDA GPU. Isaac validation needs Isaac Sim's `python.sh` configured under External tools. You can turn off Isaac QA temporarily in Config → QA.
""",
    },
    {
        "id": "glossary",
        "title": "Glossary",
        "body": """## Glossary

- **COLMAP** — Structure-from-motion: estimates camera poses from images.
- **Splat / ParticleField** — Photoreal Gaussian visual layer (pretty pictures; not the physics mesh).
- **Manifest** — `scene_manifest.json` listing objects, approvals, and artifact paths.
- **Movable** — Object that should get its own rigid mesh + physics (vs fixed room geometry).
- **Metric transform** — 4×4 COLMAP→USD similarity so the scene is upright and in meters.
- **Clean plate** — Photos of the empty room used to see surfaces behind objects.
- **USD / OpenUSD** — Universal Scene Description; Isaac Sim loads these stages.
""",
    },
    {
        "id": "capture",
        "title": "Capture tips",
        "body": """## Capture tips

- Walk slowly with **>60% overlap** between frames; cover objects from multiple heights.
- Lock exposure / white balance / focus; 720p–1080p is enough for most rooms.
- Ideal passes: full scene → **clean plate** (empty) → object close-ups → optional HDR.
- Prefer RGB-D or a known measured length for accurate meters.
- Avoid whip pans, blank walls, and people walking through the shot.
""",
    },
    {
        "id": "hybrid",
        "title": "Hybrid Scan → USD",
        "body": """## Hybrid workflow (primary)

Reconstruct → Init → Segment → Review → Align / Metric → Build → Validate.

Focused stage commands (visual, static, object, materials, lighting, package) live under **Commands** if you need to re-run one piece.
""",
    },
    {
        "id": "segmentation",
        "title": "Segmentation & review",
        "body": """## Segmentation & review

`segment-usd` writes proposals and masks. Before production continues you typically need:

- Each object marked **approved**
- Enough masks per object (`min views`)
- Enough hidden-background coverage for movable objects (or a clean plate)

Use **Review** in this GUI to edit classes, upload corrected masks, and approve gates.
""",
    },
    {
        "id": "metric",
        "title": "Metric scale",
        "body": """## Metric scale (RGB-only)

After **align-floor**, the scene is Z-up with floor at Z=0 but **unit scale** until you approve meters.

**Prefer `apply-metric-scale`** (Pipeline → Apply metric scale, or Commands):

- **Known length:** measure a real edge in meters (`--known-length-m`) and the same edge in floor-aligned / COLMAP units (`--source-length`).
- **Or** pass `--meters-per-unit` if you already know the scale factor.

Always set **reviewer** (your name) — it is recorded on the approval.

**`set-metric-transform`** is the lower-level escape hatch: supply a full 4×4 JSON yourself. Use it only when you already have a measured COLMAP→USD matrix.

After approving scale, rebuild baked meshes / re-run package so collision matches the new metric frame. See docs/USAGE.md “Metric scale”.
""",
    },
    {
        "id": "grut",
        "title": "3DGRUT / splat quality",
        "body": """## 3DGRUT & grut_overrides

Visual reconstruction uses 3DGRUT when `reconstruction.visual_backend` is `3dgrut`.

- **grut_max_iterations** — training steps. Keep LR schedule `max_steps` in sync.
- **grut_overrides** — extra Hydra `key=value` lines (Config → Advanced → Reconstruction), e.g. densify / SH / loss / `scheduler.positions.max_steps=30000`.

For legacy Splatfacto OOM: lower `splatfacto.camera_res_scale_factor` or raise `model_num_downscales`.

Full recipes: docs/USAGE.md “Environment splat quality”.
""",
    },
    {
        "id": "tools",
        "title": "Tools scripts",
        "body": """## Tools (in-repo scripts)

**Commands → Tools** runs allowlisted scripts under `tools/` through the same job log as CLI commands (no arbitrary shell):

- SAM2 runner, masked object recon, splat USD cleanup
- Isaac view / validate / convert (uses `external.isaac_python` when set)

Paths are relative to your project working directory (usually the repo root).
""",
    },
    {
        "id": "troubleshooting",
        "title": "Troubleshooting",
        "body": """## Troubleshooting (short)

- **Build exit code 2** — production review gate. Open Review, approve, run Build again.
- **RGB-only physics wrong size** — you skipped metric scale. Run **apply-metric-scale**, then rebuild.
- **Doctor red items** — install missing binaries or set paths under Config → External tools.
- **3DGRUT / CUDA OOM** — lower iterations or add quality overrides carefully; see grut section.
- **Isaac validate fails** — confirm `external.isaac_python` points at Isaac’s `python.sh`.

Deeper checklist: docs/USAGE.md → Troubleshooting.
""",
    },
    {
        "id": "legacy",
        "title": "Legacy YOLO path",
        "body": """## Legacy Splatfacto → YOLO

Still available under Pipeline's legacy strip and Commands, but **not** used for production Isaac USD:

preprocess → reconstruct → label → lift → synthesize → export-dataset → benchmark
""",
    },
]


def get_guide() -> dict[str, Any]:
    return {
        "sections": GUIDE_SECTIONS,
        "toc": [{"id": s["id"], "title": s["title"]} for s in GUIDE_SECTIONS],
    }
