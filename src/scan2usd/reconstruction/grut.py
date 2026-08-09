"""3DGRUT dataset preparation and standard ParticleField USD export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import SceneManifest
from scan2usd.reconstruction.external_cli import ExternalToolAdapter, resolve_colmap, resolve_external_command
from scan2usd.synthetic.transforms_io import find_transforms_json, load_transforms_json


@dataclass(frozen=True)
class GrutDataset:
    root: Path
    images_dir: Path
    sparse_dir: Path
    held_out_manifest: Path
    test_split_interval: int
    masked_pixels_fraction: float


def _clear_target(target: Path) -> None:
    """
    Remove an existing staged file before writing a new one.

    Not optional. Staging at ``grut_downscale: 1`` leaves symlinks pointing back
    into ``ns_data/images``; writing to such a path follows the link and
    overwrites the source capture instead of the staged copy. Switching a scene
    from 1 to 2 destroyed 928 originals exactly this way.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()


def _stage_image(source: Path, target: Path, downscale: int) -> None:
    """Copy (or link) a frame into the training set, optionally downscaled."""
    if downscale <= 1:
        _link_or_copy(source, target)
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        size = (max(1, image.width // downscale), max(1, image.height // downscale))
        resized = image.resize(size, Image.Resampling.LANCZOS)
    _clear_target(target)
    resized.save(target, quality=95)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _find_instance_mask(mask_dir: Path, image_path: Path) -> Path | None:
    candidates = (
        mask_dir / f"{image_path.stem}.png",
        mask_dir / image_path.name,
        mask_dir / f"{image_path.stem}_mask.png",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _environment_keep_mask(
    image_path: Path,
    object_masks: Sequence[Path],
) -> tuple[np.ndarray, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    keep = np.ones((height, width), dtype=np.uint8) * 255
    masked = 0
    for path in object_masks:
        with Image.open(path) as raw_mask:
            mask = np.asarray(
                raw_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
            )
        foreground = mask > 127
        newly_masked = foreground & (keep > 0)
        masked += int(newly_masked.sum())
        keep[foreground] = 0
    return keep, masked


@dataclass(frozen=True)
class _StageJob:
    """Everything one frame's staging needs, resolved up front so it can be pickled."""

    image_path: Path
    staged_image: Path
    staged_mask: Path
    object_masks: tuple[Path, ...]
    clean_plate: Path | None
    downscale: int


def _stage_one(job: _StageJob) -> tuple[int, int]:
    """Stage one frame + its keep mask. Returns (total pixels, masked pixels)."""
    keep, masked = _environment_keep_mask(job.image_path, job.object_masks)
    if job.clean_plate is not None and masked:
        with Image.open(job.image_path) as source_raw, Image.open(job.clean_plate) as clean_raw:
            source = np.asarray(source_raw.convert("RGB"))
            clean = np.asarray(
                clean_raw.convert("RGB").resize(
                    (source.shape[1], source.shape[0]),
                    Image.Resampling.LANCZOS,
                )
            )
        composited = source.copy()
        foreground = keep == 0
        composited[foreground] = clean[foreground]
        composited_image = Image.fromarray(composited)
        if job.downscale > 1:
            composited_image = composited_image.resize(
                (composited.shape[1] // job.downscale, composited.shape[0] // job.downscale),
                Image.Resampling.LANCZOS,
            )
        _clear_target(job.staged_image)
        composited_image.save(job.staged_image)
        keep[:] = 255
    else:
        _stage_image(job.image_path, job.staged_image, job.downscale)
    mask_image = Image.fromarray(keep, mode="L")
    if job.downscale > 1:
        mask_image = mask_image.resize(
            (max(1, keep.shape[1] // job.downscale), max(1, keep.shape[0] // job.downscale)),
            Image.Resampling.NEAREST,
        )
    _clear_target(job.staged_mask)
    mask_image.save(job.staged_mask)
    return int(keep.size), masked


def _stage_key(job: _StageJob) -> str:
    """
    Identify the inputs that decide a staged frame's contents.

    Size and mtime rather than a content hash: reading 928 4K JPEGs to decide
    whether to re-encode them costs about as much as re-encoding them, which
    would defeat the point. The failure mode this misses — a file rewritten
    within the same mtime granularity at exactly the same size — does not
    happen to frames written once by extraction.
    """
    parts = [str(job.downscale)]
    for path in (job.image_path, job.clean_plate, *job.object_masks):
        if path is None:
            parts.append("-")
            continue
        try:
            info = path.stat()
        except OSError:
            parts.append(f"{path}:missing")
        else:
            parts.append(f"{path}:{info.st_size}:{info.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _stage_frames(jobs: Sequence[_StageJob], *, index_path: Path) -> tuple[int, int]:
    """
    Stage every frame, skipping those already staged from identical inputs.

    Two changes over the loop this replaces, both of which matter once
    ``grut_downscale`` is above 1 and each frame is a real 4K decode + LANCZOS
    resize rather than a symlink:

    * the work runs across cores — the frames are independent, and this was the
      one stage still spending minutes on a single core;
    * a re-run with unchanged frames does nothing, because staging previously
      had no skip at all and repeated the full resize on every training run.

    The pixel counts are cached alongside the keys, so a skipped frame still
    contributes to ``masked_pixels_fraction`` without being re-read.
    """
    previous: dict[str, list[int | str]] = {}
    if index_path.is_file():
        try:
            stored = json.loads(index_path.read_text(encoding="utf-8"))
            if stored.get("schema_version") == "1.0":
                previous = stored.get("frames", {})
        except (OSError, ValueError):
            previous = {}

    index: dict[str, list[int | str]] = {}
    pending: list[_StageJob] = []
    pending_keys: list[str] = []
    for job in jobs:
        key = _stage_key(job)
        cached = previous.get(job.image_path.name)
        staged_present = (
            job.staged_image.exists() or job.staged_image.is_symlink()
        ) and job.staged_mask.is_file()
        if cached and cached[0] == key and staged_present:
            index[job.image_path.name] = cached
        else:
            pending.append(job)
            pending_keys.append(key)

    if pending:
        workers = max(1, min(len(pending), (os.cpu_count() or 2) - 2))
        if workers > 1 and len(pending) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_stage_one, pending, chunksize=4))
        else:
            results = [_stage_one(job) for job in pending]
        for job, key, (total, masked) in zip(pending, pending_keys, results, strict=True):
            index[job.image_path.name] = [key, total, masked]

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps({"schema_version": "1.0", "frames": index}, indent=2) + "\n",
        encoding="utf-8",
    )
    total_pixels = sum(int(entry[1]) for entry in index.values())
    masked_pixels = sum(int(entry[2]) for entry in index.values())
    return total_pixels, masked_pixels


def _matching_clean_plate(clean_plate_dir: Path | None, image_path: Path) -> Path | None:
    if clean_plate_dir is None:
        return None
    candidates = (
        clean_plate_dir / image_path.name,
        clean_plate_dir / f"{image_path.stem}.png",
        clean_plate_dir / f"{image_path.stem}.jpg",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def validate_background_coverage(cfg: SceneConfig, manifest: SceneManifest) -> None:
    """Block production builds that would retain ghosts or expose unknown background."""
    if manifest.build_mode != "production":
        return
    if cfg.qa.allow_background_holes:
        return
    if cfg.capture.clean_plate_dir is not None:
        return
    insufficient = [
        obj.instance_id
        for obj in manifest.objects
        if obj.movable
        and obj.review_state == "approved"
        and obj.observed_background_coverage < cfg.qa.min_background_coverage
    ]
    if insufficient:
        raise RuntimeError(
            "Production environment has unobserved background behind movable objects: "
            + ", ".join(insufficient)
            + ". Supply capture.clean_plate_dir or acquire more views "
            "(or set qa.allow_background_holes for development)."
        )


def _test_interval(ratio: float) -> int:
    if ratio <= 0:
        return 0
    return max(2, int(round(1.0 / ratio)))


def _materialize_grut_sparse(
    cfg: SceneConfig,
    source_sparse: Path,
    target_sparse: Path,
    downscale: int = 1,
) -> None:
    """
    Copy COLMAP sparse data, coerce OPENCV intrinsics to PINHOLE for 3DGRUT, and
    rescale them when the staged images were downscaled.

    Intrinsics and image size must agree. Staging half-size images against
    full-size intrinsics would put every projection in the wrong place while
    still training happily, so the two are always changed together.
    """
    model_source = source_sparse / "0"
    if not model_source.is_dir():
        raise FileNotFoundError(f"Missing COLMAP sparse model: {model_source}")
    model_target = target_sparse / "0"
    if model_target.exists():
        shutil.rmtree(model_target)
    model_target.mkdir(parents=True, exist_ok=True)

    convert_root = target_sparse / "_colmap_convert"
    if convert_root.exists():
        shutil.rmtree(convert_root)
    convert_root.mkdir(parents=True, exist_ok=True)
    txt_dir = convert_root / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    colmap = resolve_colmap(cfg)
    subprocess = ExternalToolAdapter("colmap", [colmap])
    subprocess.run(
        "model_converter",
        "--input_path",
        str(model_source.resolve()),
        "--output_path",
        str(txt_dir.resolve()),
        "--output_type",
        "TXT",
    )
    cameras_path = txt_dir / "cameras.txt"
    if cameras_path.is_file():
        lines = cameras_path.read_text(encoding="utf-8").splitlines()
        patched: list[str] = []
        for line in lines:
            if line.startswith("#") or not line.strip():
                patched.append(line)
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[1] == "OPENCV":
                parts = [parts[0], "PINHOLE", *parts[2:8]]
            if downscale > 1 and len(parts) >= 8:
                width = max(1, int(parts[2]) // downscale)
                height = max(1, int(parts[3]) // downscale)
                scaled = [f"{float(v) / downscale:.10g}" for v in parts[4:8]]
                parts = [parts[0], parts[1], str(width), str(height), *scaled, *parts[8:]]
            patched.append(" ".join(parts))
        cameras_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
    subprocess.run(
        "model_converter",
        "--input_path",
        str(txt_dir.resolve()),
        "--output_path",
        str(model_target.resolve()),
        "--output_type",
        "BIN",
    )
    shutil.rmtree(convert_root, ignore_errors=True)


def prepare_grut_dataset(
    cfg: SceneConfig,
    manifest: SceneManifest,
    *,
    output_dir: Path | None = None,
) -> GrutDataset:
    """
    Stage a COLMAP dataset with 3DGRUT's ``<stem>_mask.png`` convention.

    Approved movable-object masks are interpreted as foreground (white) and
    inverted into a static-environment keep mask (white = train, black = ignore).
    """
    root = output_dir or (cfg.workspace_dir / "build" / "grut_dataset")
    validate_background_coverage(cfg, manifest)
    # Training cost scales with pixels per iteration, not dataset size (3DGRUT
    # streams one image per step). A 4K capture is ~9x the pixels of a 720p one,
    # so staging at a reduced resolution is the main lever on training time.
    downscale = max(1, int(cfg.reconstruction.grut_downscale))
    images_out = root / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    source_images = cfg.nerfstudio_data_dir / "images"
    sparse_source = cfg.nerfstudio_data_dir / "colmap" / "sparse"
    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing Nerfstudio images: {source_images}")
    if not sparse_source.is_dir():
        raise FileNotFoundError(f"Missing COLMAP sparse model: {sparse_source}")

    object_mask_dirs = [
        Path(obj.mask_dir)
        for obj in manifest.objects
        if obj.movable and obj.review_state == "approved" and obj.mask_dir
    ]
    images = sorted(
        path
        for path in source_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise RuntimeError(f"No images found under {source_images}")

    held_out: list[str] = []
    interval = _test_interval(cfg.reconstruction.held_out_ratio)
    jobs = []
    for index, image_path in enumerate(images):
        jobs.append(
            _StageJob(
                image_path=image_path,
                staged_image=images_out / image_path.name,
                staged_mask=images_out / f"{image_path.stem}_mask.png",
                object_masks=tuple(
                    resolved
                    for resolved in (
                        _find_instance_mask(mask_dir, image_path)
                        for mask_dir in object_mask_dirs
                    )
                    if resolved is not None
                ),
                clean_plate=_matching_clean_plate(cfg.capture.clean_plate_dir, image_path),
                downscale=downscale,
            )
        )
        if interval and index % interval == 0:
            held_out.append(image_path.name)

    total_pixels, masked_pixels = _stage_frames(jobs, index_path=root / "staging_index.json")

    sparse_out = root / "sparse"
    if sparse_out.exists():
        if sparse_out.is_symlink() or sparse_out.is_file():
            sparse_out.unlink()
        else:
            shutil.rmtree(sparse_out)
    _materialize_grut_sparse(cfg, sparse_source, sparse_out, downscale)

    tjson = find_transforms_json(cfg.nerfstudio_data_dir)
    cameras: dict[str, list[list[float]]] = {}
    if tjson is not None:
        paths, matrices, _meta = load_transforms_json(tjson)
        cameras = {Path(path).name: matrix.tolist() for path, matrix in zip(paths, matrices)}
    held_out_data = {
        "schema_version": "1.0",
        "test_split_interval": interval,
        "images": [
            {"file": name, "camera_to_world": cameras.get(name)}
            for name in held_out
        ],
        "warning": (
            "3DGRUT dataset.test_split_interval excludes these views from optimization; "
            "render them from the exported camera data for unbiased comparison."
        ),
    }
    held_out_path = root / "held_out.json"
    held_out_path.write_text(json.dumps(held_out_data, indent=2) + "\n", encoding="utf-8")

    fraction = masked_pixels / max(total_pixels, 1)
    report = {
        "images": len(images),
        "downscale": downscale,
        "approved_object_mask_dirs": len(object_mask_dirs),
        "masked_pixels_fraction": fraction,
        "held_out_images": len(held_out),
    }
    (root / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return GrutDataset(
        root=root,
        images_dir=images_out,
        sparse_dir=sparse_out,
        held_out_manifest=held_out_path,
        test_split_interval=interval,
        masked_pixels_fraction=fraction,
    )


def resolve_grut(cfg: SceneConfig) -> tuple[ExternalToolAdapter, Path]:
    root_raw = str((cfg.external or {}).get("grut_root", "")).strip()
    if not root_raw:
        raise FileNotFoundError(
            "Set external.grut_root to a 3DGRUT v1.1+ checkout containing train.py"
        )
    root = Path(root_raw).expanduser().resolve()
    train = root / "train.py"
    if not train.is_file():
        raise FileNotFoundError(f"3DGRUT train.py not found under {root}")
    python = resolve_external_command(
        cfg,
        "grut_python",
        default=sys.executable,
    )
    assert python is not None
    venv_bin = Path(python[0]).parent
    env: dict[str, str] = {}
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    if not os.environ.get("CUDA_HOME"):
        for cuda_home in (Path("/usr/local/cuda"), Path("/usr/local/cuda-13"), Path("/usr/local/cuda-13.0")):
            if (cuda_home / "bin" / "nvcc").is_file():
                env["CUDA_HOME"] = str(cuda_home.resolve())
                env["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{env.get('PATH', os.environ.get('PATH', ''))}"
                break
    return ExternalToolAdapter("3dgrut", python, cwd=root, env=env), train


def consistent_strategy_overrides(iterations: int, densify_end_fraction: float) -> list[str]:
    """
    Keep densification, pruning and opacity resets on the same schedule.

    3DGRUT's defaults end densification at 15000 and pruning at 15000, while
    ``reset_density.end_iteration`` interpolates from ``densify.end_iteration``.
    Stretching only densification — which is what an override like
    ``strategy.densify.end_iteration=25000`` does — leaves opacity resets running
    for 10k iterations after the last prune that could clear what they killed.

    The bedroom's 50k run ended with 65.2% of its 2.9M Gaussians below opacity
    0.01: nearly two million dead primitives costing VRAM and iteration time, and
    exactly the faint-sheet population that reads as haze. Pruning now ends where
    densification does.
    """
    end = max(1, int(round(iterations * float(densify_end_fraction))))
    return [
        f"strategy.densify.end_iteration={end}",
        f"strategy.prune.end_iteration={end}",
        f"scheduler.positions.max_steps={iterations}",
    ]


def anti_fog_overrides(anti_fog: Any, iterations: int, densify_end: int) -> list[str]:
    """
    Train-time pressure against haze, using knobs 3DGRUT already ships.

    Photometric loss alone provably cannot remove a floater: once its blended
    colour reaches equilibrium with the background its opacity gradient vanishes,
    so it is never pushed toward transparent. Cleanup then inherits the problem,
    and post-hoc filtering has a hard ceiling — on the bedroom the safe rules got
    transmittance from 6.7% to 13.4% and then ran out, because what remains is
    genuinely carrying low-texture surfaces and cannot be cut without them.

    Both levers below act while the optimiser can still respond, replacing a
    faint sheet with a solid primitive rather than merely losing it:

    - ``loss.lambda_opacity`` penalises total opacity, so a Gaussian must earn
      the light it blocks.
    - ``strategy.prune_weight`` would drop Gaussians whose accumulated render
      contribution stays low — the accumulated-evidence pruning the floater
      literature converges on. It is **inert in the pinned 3DGRUT**:
      ``prune_gaussians_weight()`` exists but is never called, and the
      ``model.rolling_weight_contrib`` it reads is defined nowhere in the tree.
      The keys are still emitted so a version that implements it picks them up,
      but do not expect them to do anything today.
    """
    if not getattr(anti_fog, "enabled", False):
        return []
    overrides = [
        "loss.use_opacity=true",
        f"loss.lambda_opacity={anti_fog.lambda_opacity}",
    ]
    if anti_fog.lambda_scale > 0:
        overrides += ["loss.use_scale=true", f"loss.lambda_scale={anti_fog.lambda_scale}"]
    if anti_fog.prune_weight_threshold > 0:
        start = max(1, int(round(iterations * anti_fog.prune_weight_start_fraction)))
        overrides += [
            f"strategy.prune_weight.frequency={anti_fog.prune_weight_frequency}",
            f"strategy.prune_weight.start_iteration={start}",
            # Never past the last densification: pruning with nothing left to
            # replace what it removes is how the MCMC runs hollowed themselves out.
            f"strategy.prune_weight.end_iteration={densify_end}",
            f"strategy.prune_weight.weight_threshold={anti_fog.prune_weight_threshold}",
        ]
    return overrides


def _merge_overrides(generated: list[str], user: list[str]) -> list[str]:
    """User-supplied Hydra overrides win over anything generated."""
    claimed = {str(item).split("=", 1)[0].strip() for item in user if str(item).strip()}
    merged = [item for item in generated if item.split("=", 1)[0] not in claimed]
    merged += [str(item).strip() for item in user if str(item).strip()]
    return merged


def grut_train_args(
    cfg: SceneConfig,
    dataset: GrutDataset,
    *,
    output_dir: Path,
    output_usd: Path,
) -> list[str]:
    """Hydra arguments for a standard, non-NuRec ParticleField export."""
    args = [
        "train.py",
        "--config-name",
        cfg.reconstruction.grut_config,
        f"path={dataset.root.resolve()}",
        f"out_dir={output_dir.resolve()}",
        f"experiment_name={cfg.name}_static",
        f"dataset.test_split_interval={dataset.test_split_interval}",
        f"n_iterations={cfg.reconstruction.grut_max_iterations}",
        "export_usd.enabled=true",
        f"export_usd.path={output_usd.resolve()}",
        # "nurec" is Omniverse's native neural-volume format and the one NVIDIA's
        # own Isaac workflows consume; unlike the "standard" ParticleField export
        # it carries render bounds, so a scene can be clipped for inspection
        # without deleting Gaussians the observed views still need.
        f"export_usd.format={cfg.reconstruction.usd_splat_format}",
        # Preserve COLMAP coordinates; the layered scene applies the one audited
        # COLMAP→USD metric transform shared by splats, meshes, and cameras.
        "export_usd.apply_normalizing_transform=false",
        "export_usd.sorting_mode_hint=rayHitDistance",
    ]
    iterations = int(cfg.reconstruction.grut_max_iterations)
    generated: list[str] = []
    if cfg.reconstruction.grut_schedule_autofix:
        generated += consistent_strategy_overrides(
            iterations, cfg.reconstruction.densify_end_fraction
        )
    densify_end = max(
        1, int(round(iterations * float(cfg.reconstruction.densify_end_fraction)))
    )
    generated += anti_fog_overrides(cfg.reconstruction.anti_fog, iterations, densify_end)
    args.extend(_merge_overrides(generated, list(cfg.reconstruction.grut_overrides)))
    return args


def find_latest_grut_checkpoint(build_root: Path) -> Path | None:
    """Newest ``ckpt_last.pt`` under a 3DGRUT output directory."""
    checkpoints = sorted(
        build_root.rglob("ckpt_last.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return checkpoints[0] if checkpoints else None


def _recover_export_from_checkpoint(
    cfg: SceneConfig,
    build_root: Path,
    output_usd: Path,
) -> bool:
    """
    Re-run the USD export alone, from the saved checkpoint.

    A fresh process carries none of training's optimizer state, so the export
    fits in memory that the training process could not spare. Returns False when
    there is no checkpoint to recover from.
    """
    checkpoint = find_latest_grut_checkpoint(build_root)
    if checkpoint is None:
        return False
    adapter, _train = resolve_grut(cfg)
    args = [
        "-m",
        "threedgrut.export.scripts.export_usd",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output_usd.resolve()),
        "--format",
        cfg.reconstruction.usd_splat_format,
        "--no-transform",
        "--no-cameras",
    ]
    print(
        f"[scan2usd] in-process export unavailable; re-exporting from {checkpoint.name}",
        flush=True,
    )
    adapter.run(*args)
    return output_usd.is_file()


def export_environment_particlefield(
    cfg: SceneConfig,
    manifest: SceneManifest,
    *,
    dataset: GrutDataset | None = None,
) -> Path:
    """Train 3DGRUT and export the environment as standard ParticleField USD."""
    dataset = dataset or prepare_grut_dataset(cfg, manifest)
    build_root = cfg.workspace_dir / "build" / "visual"
    # NuRec always emits a USDZ container; naming it .usd produces a zip that
    # OpenUSD refuses to open. Match the extension to the format.
    suffix = "usdz" if cfg.reconstruction.usd_splat_format == "nurec" else "usd"
    output_usd = build_root / f"environment_splat.{suffix}"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    adapter, _train = resolve_grut(cfg)
    try:
        adapter.run(*grut_train_args(cfg, dataset, output_dir=build_root, output_usd=output_usd))
    except Exception as exc:  # noqa: BLE001
        # 3DGRUT exports from inside the training process, which is still holding
        # the optimizer state, so the export can run out of VRAM even though
        # training finished. Observed repeatedly on 8 GB: ~7.4 GiB in use, export
        # asks for another 0.5 GiB and dies. The checkpoint is already on disk at
        # that point, so re-export in a clean process rather than losing the run.
        if not _recover_export_from_checkpoint(cfg, build_root, output_usd):
            raise
        manifest.warnings.append(
            f"3DGRUT in-process export failed ({type(exc).__name__}); "
            "recovered by re-exporting from ckpt_last.pt"
        )
    if not output_usd.is_file():
        candidates = sorted(build_root.rglob("*.usd")) + sorted(build_root.rglob("*.usdc"))
        if len(candidates) == 1:
            output_usd = candidates[0]
        elif not _recover_export_from_checkpoint(cfg, build_root, output_usd):
            raise FileNotFoundError(
                f"3DGRUT completed but did not create configured ParticleField USD: {output_usd}"
            )

    cleanup_meta: dict = {}
    if cfg.reconstruction.splat_cleanup.enabled:
        from scan2usd.reconstruction.splat_cleanup import (
            cleanup_particlefield,
            write_report_json,
        )

        raw_backup = build_root / "environment_splat_raw.usd"
        report = cleanup_particlefield(
            cfg,
            output_usd,
            output_usd,
            cfg.reconstruction.splat_cleanup.to_params(),
            raw_backup_path=raw_backup,
        )
        report_path = build_root / "splat_cleanup_report.json"
        write_report_json(report, report_path)
        cleanup_meta = {
            "splat_cleanup": report.to_dict(),
            "splat_cleanup_report": str(report_path.resolve()),
            "raw_backup": str(raw_backup.resolve()),
        }

    manifest.register_artifact(
        artifact_id="environment_splat",
        kind="usd_particle_field",
        path=output_usd,
        producer="3dgrut",
        metadata={
            "format": "UsdVol.ParticleField3DGaussianSplat",
            "source_frame": "colmap_world",
            "masked_pixels_fraction": dataset.masked_pixels_fraction,
            "test_split_interval": dataset.test_split_interval,
            **cleanup_meta,
        },
    )
    return output_usd
