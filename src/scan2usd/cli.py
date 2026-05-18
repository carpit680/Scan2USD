from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from scan2usd.cleanup import CleanTierEnum, run_cleanup
from scan2usd.config import SceneConfig
from scan2usd.eval.benchmark import compare_abc, run_experiment
from scan2usd.export_dataset import (
    build_mixed_dataset,
    build_real_yolo_dataset,
    materialize_synthetic_train_split,
    write_synthetic_labels,
)
from scan2usd.labeling.detect import run_pseudo_labeling
from scan2usd.labeling.lift import lift_scene
from scan2usd.preprocess.video import (
    VIDEO_SUFFIXES,
    extract_frames,
    frames_dir_has_images,
    is_supported_video_suffix,
    keyframe_subsample,
)
from scan2usd.reconstruction.colmap_io import export_colmap_to_txt
from scan2usd.reconstruction.external_cli import resolve_colmap
from scan2usd.reconstruction.nerfstudio import (
    find_latest_splat_config,
    find_ns_colmap_sparse,
    ns_process_data_images,
    ns_render_camera_path,
    ns_train_splatfacto,
    ns_viewer,
)
from scan2usd.synthetic.poses import sample_novel_poses, write_nerfstudio_camera_path
from scan2usd.synthetic.transforms_io import find_transforms_json, load_transforms_json

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def preprocess(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    stride: int = typer.Option(1, help="Frame stride when extracting from video"),
    keyframe_every: int = typer.Option(1, help="Keep every Nth kept frame after blur filter"),
    max_frames: Optional[int] = typer.Option(None),
) -> None:
    """Video (MP4, MOV, …) → frames (blur filter + optional stride)."""
    cfg = SceneConfig.load(config)
    if cfg.video_path is None or not Path(cfg.video_path).exists():
        raise typer.BadParameter("config.video_path must exist for preprocess")
    vp = Path(cfg.video_path)
    if not is_supported_video_suffix(vp):
        typer.echo(
            f"Note: extension {vp.suffix!r} is not in the usual set {sorted(VIDEO_SUFFIXES)}; "
            "OpenCV will still try to decode if codecs are available.",
            err=True,
        )
    cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    paths = extract_frames(Path(cfg.video_path), cfg.frames_dir, stride=stride, max_frames=max_frames)
    paths = keyframe_subsample(paths, keyframe_every)
    typer.echo(f"Wrote {len(paths)} frames to {cfg.frames_dir}")


@app.command()
def reconstruct(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    skip_train: bool = typer.Option(False, help="Only run ns-process-data + COLMAP export"),
    skip_process_data: bool = typer.Option(
        False,
        help="Reuse existing ``nerfstudio_data_dir`` (must already contain COLMAP sparse); "
        "skip ``ns-process-data`` and only export TXT + optional train.",
    ),
    max_iterations: Optional[int] = typer.Option(
        None,
        help="Override ``splatfacto.max_num_iterations`` in YAML.",
    ),
    viewer: bool = typer.Option(
        False,
        help="Enable Nerfstudio's live Web viewer during ``ns-train`` (noisier terminal output).",
    ),
) -> None:
    """``ns-process-data`` → optional ``ns-train splatfacto``; export COLMAP TXT for lifting."""
    cfg = SceneConfig.load(config)
    if not frames_dir_has_images(cfg.frames_dir):
        if cfg.video_path and Path(cfg.video_path).is_file():
            typer.echo(f"No frames under {cfg.frames_dir}; extracting from {cfg.video_path} …")
            cfg.frames_dir.mkdir(parents=True, exist_ok=True)
            extract_frames(Path(cfg.video_path), cfg.frames_dir)
        else:
            raise typer.BadParameter(
                f"Missing or empty frames_dir: {cfg.frames_dir}. "
                "Run `scan2usd preprocess …` or set video_path to a valid file."
            )
    if skip_process_data:
        sparse = find_ns_colmap_sparse(cfg.nerfstudio_data_dir)
        if sparse is None:
            raise typer.BadParameter(
                f"--skip-process-data requires COLMAP sparse under {cfg.nerfstudio_data_dir} "
                "(e.g. colmap/sparse/0 from a prior ``ns-process-data`` run)."
            )
    else:
        ns_process_data_images(cfg, cfg.frames_dir, cfg.nerfstudio_data_dir)
        sparse = find_ns_colmap_sparse(cfg.nerfstudio_data_dir)
        if sparse is None:
            raise RuntimeError("COLMAP sparse not found after ns-process-data")
    cfg.colmap_txt_dir.mkdir(parents=True, exist_ok=True)
    export_colmap_to_txt(sparse, cfg.colmap_txt_dir, colmap_bin=resolve_colmap(cfg))
    typer.echo(f"COLMAP TXT at {cfg.colmap_txt_dir}")
    if not skip_train:
        ckpt = ns_train_splatfacto(
            cfg,
            cfg.nerfstudio_data_dir,
            max_num_iterations=max_iterations,
            enable_viewer=viewer if viewer else None,
        )
        typer.echo(f"Trained config: {ckpt}")
        typer.echo("Set splat_config_path in your YAML to this path for synthesize / view.")


@app.command()
def label(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    weights: str = typer.Option("yolov8n.pt"),
) -> None:
    """Pseudo-label frames (COCO-mapped classes) into ``workspace/labels_real``."""
    cfg = SceneConfig.load(config)
    labels_out = cfg.workspace_dir / "labels_real"
    ns_images = cfg.nerfstudio_data_dir / "images"
    if ns_images.is_dir() and any(ns_images.glob("*")):
        images_dir = ns_images
        typer.echo(f"Labeling Nerfstudio images (match COLMAP): {images_dir}")
    else:
        images_dir = cfg.frames_dir
        typer.echo(f"Labeling preprocess frames: {images_dir}")
    run_pseudo_labeling(images_dir, labels_out, cfg.classes, weights=weights)
    typer.echo(f"Labels written to {labels_out}")


@app.command()
def lift(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Lift 2D labels + COLMAP points → merged 3D AABBs (``workspace/objects_3d.npz``)."""
    cfg = SceneConfig.load(config)
    labels_dir = cfg.workspace_dir / "labels_real"
    min_p = int(cfg.lift.get("min_points_in_box", 8))
    merge_d = float(cfg.lift.get("merge_center_dist_m", 0.35))
    objs = lift_scene(cfg.colmap_txt_dir, labels_dir, min_points=min_p, merge_center_dist_m=merge_d)
    out = cfg.workspace_dir / "objects_3d.npz"
    if not objs:
        typer.echo("No 3D objects lifted; check labels and COLMAP.")
        return
    cls = np.array([o.class_id for o in objs], dtype=np.int32)
    bb = np.stack([o.bbox for o in objs], axis=0)
    np.savez(out, class_id=cls, bbox_min=bb[:, 0], bbox_max=bb[:, 1])
    typer.echo(f"Saved {len(objs)} objects to {out}")


@app.command()
def view(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    load_config: Optional[Path] = typer.Option(
        None,
        "--load-config",
        help="Path to Nerfstudio ``config.yml`` (default: ``splat_config_path`` or latest under ``ns_outputs``).",
    ),
) -> None:
    """Open the trained Splatfacto scene in Nerfstudio's Web viewer (``ns-viewer``)."""
    cfg = SceneConfig.load(config)
    ckpt = load_config if load_config is not None else find_latest_splat_config(cfg)
    if ckpt is None or not ckpt.is_file():
        raise typer.BadParameter(
            "No splat ``config.yml`` found. Run ``scan2usd reconstruct`` first, set ``splat_config_path`` "
            "in YAML, or pass ``--load-config``."
        )
    typer.echo(f"Loading {ckpt} (open the HTTP URL printed below in your browser) …")
    ns_viewer(cfg, ckpt)


@app.command("sanity-cam")
def sanity_cam(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    count: int = typer.Option(5, help="Number of real training poses to export"),
) -> None:
    """Write a Nerfstudio camera JSON for the first K real poses (manual ``ns-render`` sanity)."""
    cfg = SceneConfig.load(config)
    tjson = find_transforms_json(cfg.nerfstudio_data_dir)
    if tjson is None:
        raise RuntimeError(f"No transforms.json under {cfg.nerfstudio_data_dir}")
    _paths, mats, meta = load_transforms_json(tjson)
    if not mats:
        raise RuntimeError("No poses in transforms.json")
    from PIL import Image

    first = cfg.nerfstudio_data_dir / _paths[0].lstrip("./")
    im = Image.open(first)
    w, h = im.size
    out = cfg.workspace_dir / "camera_path_sanity_real.json"
    write_nerfstudio_camera_path(mats[:count], width=w, height=h, meta=meta, out_path=out)
    typer.echo(f"Wrote {out} (poses={min(count, len(mats))})")


@app.command()
def synthesize(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    skip_render: bool = typer.Option(False, help="Write camera path + labels only"),
) -> None:
    """Sample novel poses, write Nerfstudio camera JSON, optional ``ns-render``, synthetic YOLO labels."""
    cfg = SceneConfig.load(config)
    tjson = find_transforms_json(cfg.nerfstudio_data_dir)
    if tjson is None:
        raise RuntimeError(f"No transforms.json under {cfg.nerfstudio_data_dir}")
    paths, mats, meta = load_transforms_json(tjson)
    if not mats:
        raise RuntimeError("No camera poses in transforms.json")
    # image size from first frame file if present
    first = cfg.nerfstudio_data_dir / paths[0].lstrip("./")
    from PIL import Image

    im = Image.open(first)
    w, h = im.size
    ps = cfg.pose_sampling
    poses = sample_novel_poses(
        mats,
        num_poses=int(ps.get("num_poses", 200)),
        position_jitter_m=float(ps.get("position_jitter_m", 0.05)),
        height_jitter_m=float(ps.get("height_jitter_m", 0.02)),
        max_rotation_deg=float(ps.get("max_rotation_deg", 8.0)),
        interpolation_keyframes=int(ps.get("interpolation_keyframes", 8)),
        seed=cfg.seed,
    )
    cam_json = cfg.workspace_dir / "camera_path.json"
    write_nerfstudio_camera_path(poses, width=w, height=h, meta=meta, out_path=cam_json)
    typer.echo(f"Wrote camera path ({len(poses)} poses) to {cam_json}")

    objs_npz = cfg.workspace_dir / "objects_3d.npz"
    if objs_npz.exists():
        z = np.load(objs_npz)
        from scan2usd.labeling.lift import Object3D

        objs = [
            Object3D(int(c), np.stack([lo, hi], axis=0))
            for c, lo, hi in zip(z["class_id"], z["bbox_min"], z["bbox_max"])
        ]
        lbl_dir = cfg.workspace_dir / "labels_synthetic"
        write_synthetic_labels(objs, poses, meta, w, h, lbl_dir)
        typer.echo(f"Synthetic labels in {lbl_dir}")
    else:
        typer.echo("Skip synthetic labels: run `scan2usd lift` first.")

    if skip_render:
        return
    load_cfg = cfg.splat_config_path
    if load_cfg is None or not Path(load_cfg).exists():
        typer.echo("splat_config_path not set or missing; skip ns-render (train first or set path).")
        return
    cfg.renders_dir.mkdir(parents=True, exist_ok=True)
    ns_render_camera_path(cfg, Path(load_cfg), cam_json, cfg.renders_dir)
    typer.echo(f"Renders under {cfg.renders_dir}")


@app.command("export-dataset")
def export_dataset_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: str = typer.Option(..., help="real | synthetic | mixed"),
) -> None:
    """Build YOLO tree + data.yaml for the selected experiment mode."""
    cfg = SceneConfig.load(config)
    real_yaml = build_real_yolo_dataset(cfg)
    real_root = real_yaml.parent
    if mode == "real":
        typer.echo(real_yaml)
        return
    if mode == "synthetic":
        syn_yaml = materialize_synthetic_train_split(
            cfg,
            cfg.renders_dir,
            cfg.workspace_dir / "labels_synthetic",
            real_root_for_val=real_root,
        )
        typer.echo(syn_yaml)
        return
    if mode == "mixed":
        syn_root = (cfg.workspace_dir / "dataset_synthetic")
        if not (syn_root / "images" / "train").exists():
            materialize_synthetic_train_split(
                cfg,
                cfg.renders_dir,
                cfg.workspace_dir / "labels_synthetic",
                real_root_for_val=real_root,
                output_root=syn_root,
            )
        mixed_yaml = build_mixed_dataset(cfg, real_root, syn_root)
        typer.echo(mixed_yaml)
        return
    raise typer.BadParameter("mode must be real|synthetic|mixed")


@app.command()
def benchmark(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    experiment: str = typer.Option("all", help="A | B | C | all"),
) -> None:
    """Train/eval YOLO for experiment A/B/C (or all) and write JSON reports."""
    cfg = SceneConfig.load(config)
    reports = cfg.workspace_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    def yaml_for(ex: str) -> Path:
        if ex == "A":
            return build_real_yolo_dataset(cfg)
        if ex == "B":
            real_root = (cfg.workspace_dir / "dataset_real")
            build_real_yolo_dataset(cfg, output_root=real_root)
            return materialize_synthetic_train_split(
                cfg,
                cfg.renders_dir,
                cfg.workspace_dir / "labels_synthetic",
                real_root_for_val=real_root,
            )
        if ex == "C":
            real_root = (cfg.workspace_dir / "dataset_real")
            build_real_yolo_dataset(cfg, output_root=real_root)
            syn_root = cfg.workspace_dir / "dataset_synthetic"
            materialize_synthetic_train_split(
                cfg,
                cfg.renders_dir,
                cfg.workspace_dir / "labels_synthetic",
                real_root_for_val=real_root,
                output_root=syn_root,
            )
            return build_mixed_dataset(cfg, real_root, syn_root)
        raise typer.BadParameter("experiment must be A|B|C|all")

    ex_key = experiment.strip().lower()
    if ex_key == "all":
        exps = ["A", "B", "C"]
    elif ex_key in {"a", "b", "c"}:
        exps = [ex_key.upper()]
    else:
        raise typer.BadParameter("experiment must be A|B|C|all")
    for ex in exps:
        data_yaml = yaml_for(ex)
        run_experiment(ex, data_yaml, cfg, reports)
    if ex_key == "all":
        summary = compare_abc(reports)
        typer.echo(summary)


@app.command()
def clean(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    tier: CleanTierEnum = typer.Option(
        ...,
        "--tier",
        help="light: datasets + reports | medium: + labels/renders/camera paths | full: entire workspace",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="List paths that would be removed"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation for --tier full"),
    ultralytics_runs: bool = typer.Option(
        True,
        "--ultralytics-runs/--no-ultralytics-runs",
        help="Also remove ./runs/ (Ultralytics val outputs in cwd)",
    ),
    downloads: bool = typer.Option(
        False,
        "--downloads",
        help="Also remove yolo*.pt weight files downloaded in cwd (e.g. yolo26n.pt)",
    ),
) -> None:
    """Remove experiment artifacts for the scene workspace (see --tier)."""
    cfg = SceneConfig.load(config)
    tier_key = tier.value
    ws = cfg.workspace_dir.resolve()

    if tier_key == "full" and not dry_run and not yes:
        typer.confirm(
            f"Delete entire workspace at {ws} (and related paths outside it if configured)?",
            abort=True,
        )

    removed = run_cleanup(
        cfg,
        tier_key,
        dry_run=dry_run,
        include_ultralytics=ultralytics_runs,
        include_downloads=downloads,
    )
    if not removed:
        typer.echo(f"No matching artifacts for tier={tier_key!r}.")
        return
    verb = "Would remove" if dry_run else "Removed"
    typer.echo(f"{verb} {len(removed)} path(s) (tier={tier_key}):")
    for p in removed:
        typer.echo(f"  {p}")
    if tier_key == "full":
        typer.echo("Tip: clear or update splat_config_path in your YAML before reconstruct.")


@app.command()
def doctor(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Check system binaries, Python imports, and print Debian/Ubuntu apt hints when relevant."""
    from scan2usd.doctor_deps import print_doctor_report

    cfg = SceneConfig.load(config)
    print_doctor_report(cfg)


if __name__ == "__main__":
    app()
