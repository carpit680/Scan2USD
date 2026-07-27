from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from scan2usd.cleanup import CleanTierEnum, run_cleanup
from scan2usd.debug_lift import run_lift_debug
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
    clear_frame_images,
    extract_frames,
    frames_dir_has_images,
    is_supported_video_suffix,
    keyframe_subsample,
    list_frame_images,
)
from scan2usd.reconstruction.colmap_io import export_colmap_to_txt, parse_images_txt
from scan2usd.reconstruction.external_cli import resolve_colmap
from scan2usd.reconstruction.nerfstudio import (
    find_latest_splat_config,
    find_ns_colmap_sparse,
    ns_process_data_images,
    ns_render_camera_path,
    ns_train_splatfacto,
    ns_viewer,
    ns_viewer_with_boxes,
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
    blur_threshold: Optional[float] = typer.Option(
        None,
        "--blur-threshold",
        help="Laplacian-variance floor (default: reconstruction.blur_threshold). "
        "Raise to keep only sharp frames; lower to keep softer footage.",
    ),
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
    paths = extract_frames(
        Path(cfg.video_path),
        cfg.frames_dir,
        stride=stride,
        max_frames=max_frames,
        blur_threshold=(
            blur_threshold if blur_threshold is not None else cfg.reconstruction.blur_threshold
        ),
    )
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
    video_stride: int = typer.Option(
        15,
        "--video-stride",
        help="When auto-extracting from ``video_path`` (empty frames_dir): keep every Nth sharp frame. "
        "Use 1 for dense sampling (slow COLMAP); 15–30 is typical for 30fps walk-around video.",
    ),
    video_max_frames: int = typer.Option(
        600,
        "--video-max-frames",
        help="Cap frames when auto-extracting from video (0 = no cap). Prevents 10k+ near-duplicate "
        "frames that break COLMAP.",
    ),
    force: bool = typer.Option(
        False,
        help="Re-extract frames from video into frames_dir (overwrite existing JPEGs) and re-run "
        "ns-process-data. Requires video_path.",
    ),
    min_registration_rate: Optional[float] = typer.Option(
        None,
        "--min-registration-rate",
        help="Fail if COLMAP registers fewer than this fraction of input frames "
        "(default: reconstruction.min_registration_rate; 0 disables).",
    ),
    blur_threshold: Optional[float] = typer.Option(
        None,
        "--blur-threshold",
        help="Laplacian-variance floor when auto-extracting frames "
        "(default: reconstruction.blur_threshold; lower keeps softer frames).",
    ),
    min_features: Optional[int] = typer.Option(
        None,
        "--min-features",
        help="Drop frames with fewer than this many SIFT features — catches sharp "
        "but textureless frames (blank walls/ceilings) that cannot register. "
        "Default: reconstruction.min_frame_features.",
    ),
) -> None:
    """``ns-process-data`` → optional ``ns-train splatfacto``; export COLMAP TXT for lifting."""
    cfg = SceneConfig.load(config)
    need_extract = force or not frames_dir_has_images(cfg.frames_dir)
    if need_extract:
        if not (cfg.video_path and Path(cfg.video_path).is_file()):
            if force:
                raise typer.BadParameter(
                    "reconstruct --force requires a valid video_path to re-extract frames"
                )
            raise typer.BadParameter(
                f"Missing or empty frames_dir: {cfg.frames_dir}. "
                "Run `scan2usd preprocess …` or set video_path to a valid file."
            )
        cap = None if video_max_frames == 0 else video_max_frames
        cfg.frames_dir.mkdir(parents=True, exist_ok=True)
        if force and frames_dir_has_images(cfg.frames_dir):
            removed = clear_frame_images(cfg.frames_dir)
            typer.echo(f"Force: cleared {removed} existing frames under {cfg.frames_dir}")
        typer.echo(
            f"{'Re-extracting' if force else 'Extracting'} from {cfg.video_path} "
            f"(stride={video_stride}, max_frames={cap}) …"
        )
        extract_frames(
            Path(cfg.video_path),
            cfg.frames_dir,
            stride=video_stride,
            max_frames=cap,
            blur_threshold=(
                blur_threshold
                if blur_threshold is not None
                else cfg.reconstruction.blur_threshold
            ),
            min_features=(
                min_features
                if min_features is not None
                else cfg.reconstruction.min_frame_features
            ),
            max_dim=cfg.reconstruction.frame_max_dim,
        )
    if skip_process_data:
        if force:
            raise typer.BadParameter("Cannot combine --force with --skip-process-data")
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

    registered = len(parse_images_txt(cfg.colmap_txt_dir / "images.txt"))
    candidates = len(list_frame_images(cfg.frames_dir))
    rate = registered / candidates if candidates else 0.0
    typer.echo(f"COLMAP registered {registered}/{candidates} frames ({rate:.0%})")
    threshold = (
        min_registration_rate
        if min_registration_rate is not None
        else cfg.reconstruction.min_registration_rate
    )
    if threshold > 0 and rate < threshold:
        raise typer.BadParameter(
            f"COLMAP registered only {registered}/{candidates} frames ({rate:.0%}), below the "
            f"{threshold:.0%} minimum. Everything downstream (floor plane, splat, collision "
            "geometry) would be fit to these few views and look built but be wrong. "
            "Usual causes: motion blur, too-large gaps between frames, textureless surfaces. "
            "Try a denser/sharper extraction (e.g. --video-stride 8 --blur-threshold 100), "
            "or recapture with slower motion and locked exposure/focus. "
            "Pass --min-registration-rate 0 to proceed anyway."
        )
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
    lift_cfg = cfg.lift or {}
    min_p = int(lift_cfg.get("min_points_in_box", 8))
    merge_d = float(lift_cfg.get("merge_center_dist_m", 0.6))
    depth_mad = float(lift_cfg.get("depth_trim_mad", 3.0))
    inner_margin = float(lift_cfg.get("inner_margin_frac", 0.12))
    dp = lift_cfg.get("depth_percentile", [10.0, 90.0])
    depth_percentile = None if dp is None else (float(dp[0]), float(dp[1]))
    pct = lift_cfg.get("percentile", [5.0, 95.0])
    percentile = (float(pct[0]), float(pct[1]))
    max_ext = lift_cfg.get("max_extent_m")
    max_extent_m = None if max_ext is None else float(max_ext)
    objs = lift_scene(
        cfg.colmap_txt_dir,
        labels_dir,
        min_points=min_p,
        merge_center_dist_m=merge_d,
        ns_data_dir=cfg.nerfstudio_data_dir,
        depth_trim_mad=depth_mad,
        inner_margin_frac=inner_margin,
        depth_percentile=depth_percentile,
        percentile=percentile,
        max_extent_m=max_extent_m,
    )
    out = cfg.workspace_dir / "objects_3d.npz"
    if not objs:
        typer.echo("No 3D objects lifted; check labels and COLMAP.")
        return
    cls = np.array([o.class_id for o in objs], dtype=np.int32)
    bb = np.stack([o.bbox for o in objs], axis=0)
    np.savez(
        out,
        class_id=cls,
        bbox_min=bb[:, 0],
        bbox_max=bb[:, 1],
        obb_center=np.stack([o.center for o in objs], axis=0),
        obb_half=np.stack([o.half_extents for o in objs], axis=0),
        obb_rotation=np.stack([o.rotation for o in objs], axis=0),
        coord_frame=np.array("nerfstudio"),
    )
    typer.echo(f"Saved {len(objs)} objects to {out} (transforms.json / sparse_pc frame)")


@app.command("debug-lift")
def debug_lift(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        help="Output folder (default: ``<workspace>/lift_debug``).",
    ),
    max_frames: int = typer.Option(16, help="Max labeled frames to export overlays for."),
    frame_stride: int = typer.Option(
        25,
        help="Step through ``transforms.json`` frames (higher = fewer images).",
    ),
) -> None:
    """
    Debug 2D→3D lift: side-by-side YOLO (left) vs per-detection 3D lift (right).

    Use this to see whether orientation errors come from lift geometry or from the splat viewer transform.
    """
    cfg = SceneConfig.load(config)
    out = out_dir if out_dir is not None else cfg.workspace_dir / "lift_debug"
    lift_cfg = cfg.lift or {}
    dp = lift_cfg.get("depth_percentile", [10.0, 90.0])
    depth_percentile = None if dp is None else (float(dp[0]), float(dp[1]))
    path = run_lift_debug(
        cfg,
        out,
        max_frames=max_frames,
        frame_stride=frame_stride,
        min_points=int(lift_cfg.get("min_points_in_box", 8)),
        depth_trim_mad=float(lift_cfg.get("depth_trim_mad", 3.0)),
        inner_margin_frac=float(lift_cfg.get("inner_margin_frac", 0.12)),
        depth_percentile=depth_percentile,
    )
    typer.echo(f"Wrote lift debug artifacts to {path}")
    typer.echo(f"  Open images in {path} and read {path / 'summary.json'} (median IoU in summary).")


@app.command()
def view(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    load_config: Optional[Path] = typer.Option(
        None,
        "--load-config",
        help="Path to Nerfstudio ``config.yml`` (default: ``splat_config_path`` or latest under ``ns_outputs``).",
    ),
    show_boxes: bool = typer.Option(
        True,
        "--boxes/--no-boxes",
        help="Overlay lifted 3D AABBs from ``workspace/objects_3d.npz`` (run ``scan2usd lift`` first).",
    ),
    objects_npz: Optional[Path] = typer.Option(
        None,
        "--objects-npz",
        help="Path to ``objects_3d.npz`` (default: ``<workspace>/objects_3d.npz``).",
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
    npz = objects_npz if objects_npz is not None else cfg.workspace_dir / "objects_3d.npz"
    if show_boxes:
        if not npz.is_file():
            typer.echo(
                f"No {npz}; open viewer without boxes (run ``scan2usd lift`` first, or use ``--no-boxes``).",
                err=True,
            )
            show_boxes = False
        else:
            typer.echo(f"Loading {ckpt} with {len(np.load(npz)['class_id'])} 3D boxes …")
            ns_viewer_with_boxes(cfg, ckpt, npz)
            return
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

        if "obb_center" in z:
            objs = [
                Object3D(
                    int(c),
                    np.stack([lo, hi], axis=0),
                    center=ctr,
                    rotation=rot,
                    half_extents=half,
                    points=np.empty((0, 3)),
                    view_origin=ctr,
                )
                for c, lo, hi, ctr, rot, half in zip(
                    z["class_id"],
                    z["bbox_min"],
                    z["bbox_max"],
                    z["obb_center"],
                    z["obb_rotation"],
                    z["obb_half"],
                )
            ]
        else:
            objs = [
                Object3D(
                    int(c),
                    np.stack([lo, hi], axis=0),
                    center=(lo + hi) / 2,
                    rotation=np.eye(3),
                    half_extents=(hi - lo) / 2,
                    points=np.empty((0, 3)),
                    view_origin=(lo + hi) / 2,
                )
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


@app.command("init-usd")
def init_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: str = typer.Option("production", help="production | preview"),
) -> None:
    """Create the versioned scene manifest and resumable USD build state."""
    from scan2usd.pipeline.orchestrator import PipelineOrchestrator

    cfg = SceneConfig.load(config)
    orchestrator = PipelineOrchestrator(cfg, config, build_mode=mode)
    typer.echo(orchestrator.manifest_path)


@app.command("segment-usd")
def segment_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: str = typer.Option("production", help="production | preview"),
    force: bool = typer.Option(False, help="Re-run proposal and mask stages"),
) -> None:
    """Propose object instances and run the configured SAM2 mask propagator."""
    from scan2usd.pipeline.orchestrator import PipelineOrchestrator

    cfg = SceneConfig.load(config)
    orchestrator = PipelineOrchestrator(cfg, config, build_mode=mode)
    orchestrator.segment(force=force)
    typer.echo(f"Masks ready; review {orchestrator.manifest_path}")


@app.command("review")
def review_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    share: bool = typer.Option(False, help="Ask Gradio for a public share URL"),
) -> None:
    """Review/correct masks and approve objects, generated assets, and lighting."""
    from scan2usd.review.app import launch_review_app

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"Missing {manifest_path}; run scan2usd init-usd first")
    launch_review_app(cfg, manifest_path, share=share)


@app.command("align-floor")
def align_floor(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: str = typer.Option("preview", help="production | preview"),
    force: bool = typer.Option(False, help="Re-estimate even if a transform already exists"),
    distance_thresh: float = typer.Option(
        0.04,
        help="RANSAC inlier distance in COLMAP units",
    ),
) -> None:
    """
    Estimate the floor plane from COLMAP points and build COLMAP→USD alignment.

    Rotates so the floor normal is +Z and translates so the floor is at Z=0.
    Scale stays 1.0 until a metric length is approved via apply-metric-scale
    or set-metric-transform.
    Run this after reconstruct / before build-usd packaging (build-usd also runs it).
    """
    from scan2usd.pipeline.orchestrator import PipelineOrchestrator

    cfg = SceneConfig.load(config)
    orchestrator = PipelineOrchestrator(cfg, config, build_mode=mode)
    out = orchestrator.align_floor(force=force, distance_thresh=distance_thresh)
    transform = next(
        (
            item
            for item in orchestrator.manifest.transforms
            if item.source_frame == "colmap_world" and item.target_frame == "usd_world_z_up_meters"
        ),
        None,
    )
    meta = orchestrator.manifest.artifact("floor_alignment")
    if meta and meta.metadata:
        typer.echo(
            "Floor inliers="
            f"{meta.metadata.get('inliers')}/{meta.metadata.get('point_count')} "
            f"({float(meta.metadata.get('inlier_ratio', 0)):.1%})"
        )
    if transform is not None:
        typer.echo(f"Confidence={transform.confidence:.3f} evidence={transform.evidence}")
    typer.echo(f"Wrote {out}")


@app.command("set-metric-transform")
def set_metric_transform(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    transform_json: Path = typer.Argument(..., exists=True, dir_okay=False),
    reviewer: str = typer.Option(..., help="Person approving this metric registration"),
) -> None:
    """Approve a COLMAP→USD 4×4 similarity transform (Z-up, meters)."""
    import json

    from scan2usd.geometry.frames import (
        FRAME_COLMAP,
        FRAME_USD,
        uniform_scale,
        validate_similarity,
    )
    from scan2usd.pipeline.manifest import ScaleEvidence, SceneManifest, TransformRecord

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    raw = json.loads(transform_json.read_text(encoding="utf-8"))
    matrix = raw.get("colmap_to_usd") if isinstance(raw, dict) else raw
    confidence = float(raw.get("registration_confidence", 1.0)) if isinstance(raw, dict) else 1.0
    value = validate_similarity(matrix)
    scale = uniform_scale(value)
    manifest.transforms = [
        item
        for item in manifest.transforms
        if not (
            item.source_frame == FRAME_COLMAP and item.target_frame == FRAME_USD
        )
    ]
    manifest.transforms.append(
        TransformRecord(
            source_frame=FRAME_COLMAP,
            target_frame=FRAME_USD,
            matrix=value.tolist(),
            confidence=confidence,
            evidence=str(transform_json.resolve()),
        )
    )
    manifest.scale = ScaleEvidence(
        method="approved_similarity_registration",
        meters_per_source_unit=scale,
        confidence=confidence,
        reference=str(transform_json.resolve()),
        approved=True,
    )
    manifest.approve("metric_transform", reviewer=reviewer)
    manifest.save(manifest_path)
    typer.echo(f"Approved scale: {scale:.9g} meters/source-unit")


@app.command("apply-metric-scale")
def apply_metric_scale(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    reviewer: str = typer.Option(..., help="Person approving this metric registration"),
    meters_per_unit: float | None = typer.Option(
        None,
        help="Meters per COLMAP/source unit (after floor alignment)",
    ),
    known_length_m: float | None = typer.Option(
        None,
        help="Real-world length in meters of a measured edge",
    ),
    source_length: float | None = typer.Option(
        None,
        help="Same edge length in current floor-aligned scene/COLMAP units",
    ),
    floor_json: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional floor COLMAP→USD JSON (default: workspace/colmap_to_usd_floor.json)",
    ),
    output: Path | None = typer.Option(
        None,
        help="Where to write the metric transform JSON (default: workspace/colmap_to_usd_metric.json)",
    ),
) -> None:
    """
    Compose uniform metric scale onto the floor COLMAP→USD transform and approve it.

    Provide either --meters-per-unit, or both --known-length-m and --source-length.
    Rebuild baked meshes and re-run package-usd after approving so collision matches.
    """
    from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD, uniform_scale
    from scan2usd.geometry.metric_scale import (
        apply_uniform_metric_scale,
        load_colmap_to_usd_matrix,
        meters_per_unit_from_lengths,
        resolve_floor_transform_path,
        write_metric_transform_json,
    )
    from scan2usd.pipeline.manifest import ScaleEvidence, SceneManifest, TransformRecord

    if meters_per_unit is not None and (known_length_m is not None or source_length is not None):
        raise typer.BadParameter("Use either --meters-per-unit or --known-length-m/--source-length")
    if meters_per_unit is None:
        if known_length_m is None or source_length is None:
            raise typer.BadParameter(
                "Provide --meters-per-unit, or both --known-length-m and --source-length"
            )
        meters_per_unit = meters_per_unit_from_lengths(
            known_length_m=known_length_m,
            source_length=source_length,
        )
    if meters_per_unit <= 0:
        raise typer.BadParameter("--meters-per-unit must be positive")

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"Missing {manifest_path}; run scan2usd init-usd first")
    manifest = SceneManifest.load(manifest_path)

    source_path = (
        floor_json
        if floor_json is not None
        else resolve_floor_transform_path(cfg.workspace_dir, manifest.transforms)
    )
    base = load_colmap_to_usd_matrix(source_path)
    metric = apply_uniform_metric_scale(base, meters_per_unit)
    out_path = output or (cfg.workspace_dir / "colmap_to_usd_metric.json")
    if known_length_m is not None and source_length is not None:
        evidence = (
            f"known_length_m={known_length_m}; source_length={source_length}; "
            f"floor={source_path.resolve()}"
        )
        method = "floor_alignment_plus_measured_length"
    else:
        evidence = f"meters_per_unit={meters_per_unit}; floor={source_path.resolve()}"
        method = "floor_alignment_plus_meters_per_unit"

    write_metric_transform_json(
        metric,
        out_path,
        meters_per_source_unit=meters_per_unit,
        method=method,
        evidence=evidence,
        confidence=0.9,
    )

    scale = uniform_scale(metric)
    manifest.transforms = [
        item
        for item in manifest.transforms
        if not (item.source_frame == FRAME_COLMAP and item.target_frame == FRAME_USD)
    ]
    manifest.transforms.append(
        TransformRecord(
            source_frame=FRAME_COLMAP,
            target_frame=FRAME_USD,
            matrix=metric.tolist(),
            confidence=0.9,
            evidence=str(out_path.resolve()),
        )
    )
    manifest.scale = ScaleEvidence(
        method=method,
        meters_per_source_unit=scale,
        confidence=0.9,
        reference=str(out_path.resolve()),
        approved=True,
    )
    manifest.approve("metric_transform", reviewer=reviewer)
    manifest.save(manifest_path)
    typer.echo(f"Wrote {out_path}")
    typer.echo(f"Approved scale: {scale:.9g} meters/source-unit")
    typer.echo("Rebuild baked geometry (objects/static) and package-usd so meshes match the new T.")


@app.command("build-visual-usd")
def build_visual_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Build the masked static 3DGRUT ParticleField layer only."""
    from scan2usd.pipeline.manifest import SceneManifest
    from scan2usd.reconstruction.grut import export_environment_particlefield

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    output = export_environment_particlefield(cfg, manifest)
    manifest.save(manifest_path)
    typer.echo(output)


@app.command("cleanup-splat")
def cleanup_splat_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: str = typer.Option("preview", help="production | preview"),
    force: bool = typer.Option(
        False,
        help="Re-run even if a cleanup report already exists (uses raw backup)",
    ),
    outlier_std: float | None = typer.Option(
        None,
        help="Override reconstruction.splat_cleanup.outlier_std for this run",
    ),
    min_opacity: float | None = typer.Option(
        None,
        help="Override reconstruction.splat_cleanup.min_opacity for this run",
    ),
) -> None:
    """
    Remove stray Gaussians from the environment ParticleField without retraining.

    Uses reconstruction.splat_cleanup thresholds (outlier_std is the main knob).
    Re-runs from environment_splat_raw.usd when present so you can tune thresholds.
    """
    from scan2usd.pipeline.orchestrator import PipelineOrchestrator

    cfg = SceneConfig.load(config)
    if outlier_std is not None:
        cfg.reconstruction.splat_cleanup.outlier_std = float(outlier_std)
    if min_opacity is not None:
        cfg.reconstruction.splat_cleanup.min_opacity = float(min_opacity)
    cfg.reconstruction.splat_cleanup.enabled = True
    orchestrator = PipelineOrchestrator(cfg, config, build_mode=mode)
    path = orchestrator.cleanup_splat(force=force)
    report = cfg.workspace_dir / "build" / "visual" / "splat_cleanup_report.json"
    if report.is_file():
        import json

        payload = json.loads(report.read_text(encoding="utf-8"))
        typer.echo(
            "kept={kept}/{total} removed_spatial={spatial} removed_opacity={opacity} "
            "outlier_std={std}".format(
                kept=payload.get("kept_count"),
                total=payload.get("input_count"),
                spatial=payload.get("removed_spatial"),
                opacity=payload.get("removed_opacity"),
                std=payload.get("outlier_std"),
            )
        )
    typer.echo(path)


@app.command("build-static-usd")
def build_static_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Build metric static collision and shadow/depth proxy meshes only."""
    from scan2usd.geometry.static_scene import build_static_scene
    from scan2usd.pipeline.manifest import SceneManifest

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    outputs = build_static_scene(cfg, manifest, manifest_path=manifest_path)
    manifest.save(manifest_path)
    typer.echo(outputs)


@app.command("build-object-usd")
def build_object_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    instance_id: str = typer.Argument(...),
) -> None:
    """Reconstruct one approved rigid object and its collision source."""
    from scan2usd.assets.object_builder import reconstruct_object
    from scan2usd.pipeline.manifest import SceneManifest

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    outputs = reconstruct_object(cfg, manifest, instance_id)
    manifest.save(manifest_path)
    typer.echo(outputs)


@app.command("build-materials-usd")
def build_materials_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    instance_id: str = typer.Argument(...),
) -> None:
    """Build baked and PBR material variants for one reconstructed object."""
    from scan2usd.assets.materials import build_object_materials
    from scan2usd.pipeline.manifest import SceneManifest

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    outputs = build_object_materials(cfg, manifest, manifest.get_object(instance_id))
    manifest.save(manifest_path)
    typer.echo(outputs)


@app.command("build-lighting-usd")
def build_lighting_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Estimate the reviewed RTX dome-light layer."""
    from scan2usd.lighting.estimate import estimate_scene_lighting
    from scan2usd.pipeline.manifest import SceneManifest

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    estimate, output = estimate_scene_lighting(cfg, manifest)
    manifest.save(manifest_path)
    typer.echo(f"{output} ({estimate.source}, confidence={estimate.confidence:.2f})")


@app.command("package-usd")
def package_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Compose existing approved artifacts into the layered Isaac scene."""
    from scan2usd.pipeline.manifest import SceneManifest
    from scan2usd.usd.package import build_usd_package

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    root = build_usd_package(cfg, manifest)
    manifest.save(manifest_path)
    typer.echo(root)


@app.command("build-usd")
def build_usd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: str = typer.Option("production", help="production | preview"),
    force: bool = typer.Option(False, help="Re-run completed stages"),
) -> None:
    """Run/resume the complete hybrid Scan-to-USD build graph."""
    from scan2usd.pipeline.orchestrator import PipelineOrchestrator, ReviewRequired

    cfg = SceneConfig.load(config)
    orchestrator = PipelineOrchestrator(cfg, config, build_mode=mode)
    try:
        root = orchestrator.build(force=force)
    except ReviewRequired as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    typer.echo(root)


@app.command("validate-usd")
def validate_usd_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    held_out_renders: Optional[Path] = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Isaac renders for cameras in build/grut_dataset/held_out.json "
        "(default: build/heldout_renders when present — run render-heldout first)",
    ),
    isaac: bool = typer.Option(True, "--isaac/--no-isaac", help="Run headless Isaac physics tests"),
) -> None:
    """Run visual, registration, mesh, collision, and Isaac physics quality gates."""
    from scan2usd.pipeline.manifest import SceneManifest
    from scan2usd.usd.validate import validate_usd

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    if held_out_renders is None:
        default_renders = cfg.workspace_dir / "build" / "heldout_renders"
        if default_renders.is_dir():
            held_out_renders = default_renders
            typer.echo(f"Using held-out renders from {default_renders}")
    try:
        report = validate_usd(
            cfg,
            manifest,
            held_out_render_dir=held_out_renders,
            run_isaac=isaac,
        )
    finally:
        manifest.save(manifest_path)
    typer.echo(f"usable={report['usable']} report={cfg.usd.output_dir / 'build_report.json'}")


@app.command("render-heldout")
def render_heldout(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Optional[Path] = typer.Option(
        None,
        help="Render output dir (default: <workspace>/build/heldout_renders)",
    ),
    evaluate: bool = typer.Option(
        True,
        "--evaluate/--no-evaluate",
        help="Compute PSNR/SSIM/LPIPS + scene_quality.json after rendering",
    ),
    lpips: bool = typer.Option(
        True,
        "--lpips/--no-lpips",
        help="Include LPIPS (needs torchmetrics; slower)",
    ),
) -> None:
    """
    Render the packaged USD in headless Isaac at held-out capture cameras.

    Compares the real scene the camera saw against what Isaac renders — the
    ground-truth photorealism metric. Requires external.isaac_python, a packaged
    root USD, and build/grut_dataset/held_out.json.
    """
    from scan2usd.eval.scene_quality import build_scene_quality_report
    from scan2usd.pipeline.manifest import SceneManifest
    from scan2usd.reconstruction.external_cli import ExternalToolAdapter, resolve_external_command

    cfg = SceneConfig.load(config)
    manifest_path = cfg.workspace_dir / "scene_manifest.json"
    manifest = SceneManifest.load(manifest_path)
    root_artifact = manifest.artifact("root_usd")
    if root_artifact is None or not Path(root_artifact.path).is_file():
        raise typer.BadParameter("No packaged root USD; run build-usd / package-usd first")
    heldout_spec = cfg.workspace_dir / "build" / "grut_dataset" / "held_out.json"
    if not heldout_spec.is_file():
        raise typer.BadParameter(
            f"Missing {heldout_spec}; run build-visual-usd (3DGRUT dataset staging) first"
        )
    if not (cfg.colmap_txt_dir / "images.txt").is_file():
        raise typer.BadParameter(
            f"Missing COLMAP TXT under {cfg.colmap_txt_dir}; run reconstruct first"
        )
    prefix = resolve_external_command(cfg, "isaac_python", default="python.sh", required=True)
    assert prefix is not None
    out_dir = output or (cfg.workspace_dir / "build" / "heldout_renders")
    script = Path(__file__).resolve().parents[2] / "tools" / "isaac" / "render_heldout.py"
    adapter = ExternalToolAdapter("isaac_python", prefix)
    adapter.run(
        str(script),
        "--stage",
        str(Path(root_artifact.path).resolve()),
        "--held-out",
        str(heldout_spec.resolve()),
        "--colmap-txt",
        str(cfg.colmap_txt_dir.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--output",
        str(out_dir.resolve()),
    )
    typer.echo(f"Renders under {out_dir}")
    if evaluate:
        report = build_scene_quality_report(cfg, render_dir=out_dir, compute_lpips=lpips)
        photo = report["photorealism"]
        typer.echo(
            f"quality_score={report['quality_score']} "
            f"psnr={photo['mean_psnr']} ssim={photo['mean_ssim']} lpips={photo['mean_lpips']} "
            f"({photo['evaluated_views']}/{photo['expected_views']} views)"
        )


@app.command("quality-report")
def quality_report(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    lpips: bool = typer.Option(True, "--lpips/--no-lpips", help="Include LPIPS if available"),
) -> None:
    """Recompute scene_quality.json from artifacts on disk (no rendering)."""
    from scan2usd.eval.scene_quality import build_scene_quality_report

    cfg = SceneConfig.load(config)
    report = build_scene_quality_report(cfg, compute_lpips=lpips)
    out = Path(cfg.usd.output_dir or cfg.workspace_dir / "usd") / "scene_quality.json"
    typer.echo(f"quality_score={report['quality_score']} report={out}")


@app.command("tune")
def tune(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    cheap_trials: Optional[int] = typer.Option(
        None, help="Override tuning.max_cheap_trials (splat-cleanup sweep, no retrain)"
    ),
    retrain_trials: Optional[int] = typer.Option(
        None,
        help="Override tuning.max_retrain_trials (3DGRUT retrains — hours per trial)",
    ),
    promote: bool = typer.Option(
        True,
        "--promote/--no-promote",
        help="Write the winner to <config>_tuned.yaml when finished",
    ),
    lpips: Optional[bool] = typer.Option(
        None, "--lpips/--no-lpips", help="Override tuning.lpips"
    ),
) -> None:
    """
    Auto-tune the scene config against the Isaac photorealism quality score.

    Loop: adjust parameters → re-export/package the USD → render held-out views
    in Isaac → score → next trial. Resumable: completed trials are read from
    workspace/tuning/trials.json and skipped. Requires a packaged scene,
    environment_splat_raw.usd, held_out.json, and external.isaac_python.
    """
    from scan2usd.tuning.runner import run_tuning

    cfg = SceneConfig.load(config)
    raw_splat = cfg.workspace_dir / "build" / "visual" / "environment_splat_raw.usd"
    if not raw_splat.is_file():
        raise typer.BadParameter(
            f"Missing {raw_splat}; run build-usd (with splat_cleanup enabled) first"
        )
    summary = run_tuning(
        cfg,
        config.resolve(),
        max_cheap_trials=cheap_trials
        if cheap_trials is not None
        else cfg.tuning.max_cheap_trials,
        max_retrain_trials=retrain_trials
        if retrain_trials is not None
        else cfg.tuning.max_retrain_trials,
        cheap_space=cfg.tuning.cheap_params or None,
        retrain_space=cfg.tuning.retrain_params or None,
        compute_lpips=lpips if lpips is not None else cfg.tuning.lpips,
        promote=promote,
        log=typer.echo,
    )
    typer.echo(
        f"best={summary['best_trial']} score={summary['best_score']} "
        f"params={summary['best_params']}"
    )
    if summary["promoted_config"]:
        typer.echo(f"Promoted config: {summary['promoted_config']}")
    typer.echo(f"Trials: {summary['trials_json']}")


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
