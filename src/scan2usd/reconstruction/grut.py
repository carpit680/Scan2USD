"""3DGRUT dataset preparation and standard ParticleField USD export."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

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


def _stage_image(source: Path, target: Path, downscale: int) -> None:
    """Copy (or link) a frame into the training set, optionally downscaled."""
    if downscale <= 1:
        _link_or_copy(source, target)
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        size = (max(1, image.width // downscale), max(1, image.height // downscale))
        image.resize(size, Image.Resampling.LANCZOS).save(target, quality=95)


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
    object_mask_dirs: list[Path],
) -> tuple[np.ndarray, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    keep = np.ones((height, width), dtype=np.uint8) * 255
    masked = 0
    for mask_dir in object_mask_dirs:
        path = _find_instance_mask(mask_dir, image_path)
        if path is None:
            continue
        with Image.open(path) as raw_mask:
            mask = np.asarray(
                raw_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
            )
        foreground = mask > 127
        newly_masked = foreground & (keep > 0)
        masked += int(newly_masked.sum())
        keep[foreground] = 0
    return keep, masked


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

    total_pixels = 0
    masked_pixels = 0
    held_out: list[str] = []
    interval = _test_interval(cfg.reconstruction.held_out_ratio)
    for index, image_path in enumerate(images):
        staged_image = images_out / image_path.name
        keep, masked = _environment_keep_mask(image_path, object_mask_dirs)
        clean_plate = _matching_clean_plate(cfg.capture.clean_plate_dir, image_path)
        if clean_plate is not None and masked:
            with Image.open(image_path) as source_raw, Image.open(clean_plate) as clean_raw:
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
            if downscale > 1:
                composited_image = composited_image.resize(
                    (composited.shape[1] // downscale, composited.shape[0] // downscale),
                    Image.Resampling.LANCZOS,
                )
            composited_image.save(staged_image)
            keep[:] = 255
        else:
            _stage_image(image_path, staged_image, downscale)
        mask_image = Image.fromarray(keep, mode="L")
        if downscale > 1:
            mask_image = mask_image.resize(
                (max(1, keep.shape[1] // downscale), max(1, keep.shape[0] // downscale)),
                Image.Resampling.NEAREST,
            )
        mask_image.save(images_out / f"{image_path.stem}_mask.png")
        total_pixels += int(keep.size)
        masked_pixels += masked
        if interval and index % interval == 0:
            held_out.append(image_path.name)

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
    for override in cfg.reconstruction.grut_overrides:
        text = str(override).strip()
        if text:
            args.append(text)
    return args


def export_environment_particlefield(
    cfg: SceneConfig,
    manifest: SceneManifest,
    *,
    dataset: GrutDataset | None = None,
) -> Path:
    """Train 3DGRUT and export the environment as standard ParticleField USD."""
    dataset = dataset or prepare_grut_dataset(cfg, manifest)
    build_root = cfg.workspace_dir / "build" / "visual"
    output_usd = build_root / "environment_splat.usd"
    output_usd.parent.mkdir(parents=True, exist_ok=True)
    adapter, _train = resolve_grut(cfg)
    adapter.run(*grut_train_args(cfg, dataset, output_dir=build_root, output_usd=output_usd))
    if not output_usd.is_file():
        candidates = sorted(build_root.rglob("*.usd")) + sorted(build_root.rglob("*.usdc"))
        if len(candidates) == 1:
            output_usd = candidates[0]
        else:
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
