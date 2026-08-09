from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest
from scan2usd.reconstruction.grut import grut_train_args, prepare_grut_dataset


def _write_minimal_colmap_sparse(sparse_dir: Path, *, image_names: list[str]) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = sparse_dir.parent / "_colmap_txt"
    if txt_dir.exists():
        import shutil

        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True)
    txt_dir.joinpath("cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "# Number of cameras: 1",
                "1 PINHOLE 20 10 10.0 10.0 10.0 5.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(image_names)}",
    ]
    for index, name in enumerate(image_names, start=1):
        image_lines.append(
            f"{index} 1 0 0 0 0 0 0 1 {name}"
        )
        image_lines.append("")
    txt_dir.joinpath("images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    txt_dir.joinpath("points3D.txt").write_text(
        "\n".join(
            [
                "# 3D point list with one line of data per point:",
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
                "# Number of points: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    import subprocess

    subprocess.run(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(txt_dir),
            "--output_path",
            str(sparse_dir),
            "--output_type",
            "BIN",
        ],
        check=True,
    )
    import shutil

    shutil.rmtree(txt_dir, ignore_errors=True)


def _scene(tmp_path):
    ns = tmp_path / "ns_data"
    images = ns / "images"
    sparse = ns / "colmap" / "sparse" / "0"
    images.mkdir(parents=True)
    image_names = []
    for i in range(4):
        name = f"frame_{i:03d}.jpg"
        image_names.append(name)
        Image.new("RGB", (20, 10), color=(i * 20, 100, 80)).save(images / name)
    _write_minimal_colmap_sparse(sparse, image_names=image_names)
    (ns / "transforms.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": f"./images/frame_{i:03d}.jpg",
                        "transform_matrix": np.eye(4).tolist(),
                    }
                    for i in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = SceneConfig()
    cfg.name = "room"
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.nerfstudio_data_dir = ns
    cfg.reconstruction.held_out_ratio = 0.25
    return cfg


def test_prepare_grut_dataset_writes_inverted_object_masks(tmp_path):
    cfg = _scene(tmp_path)
    mask_dir = tmp_path / "masks" / "chair_001"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[:, :5] = 255
    Image.fromarray(mask).save(mask_dir / "frame_000.png")

    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    manifest.objects.append(
        ObjectRecord(
            instance_id="chair_001",
            display_name="Chair",
            class_name="chair",
            review_state="approved",
            mask_dir=str(mask_dir),
        )
    )
    dataset = prepare_grut_dataset(cfg, manifest)
    keep = np.asarray(Image.open(dataset.images_dir / "frame_000_mask.png"))
    assert np.all(keep[:, :5] == 0)
    assert np.all(keep[:, 5:] == 255)
    assert dataset.test_split_interval == 4
    held_out = json.loads(dataset.held_out_manifest.read_text())
    assert held_out["images"][0]["file"] == "frame_000.jpg"


def test_grut_args_request_standard_particle_field_by_default(tmp_path):
    cfg = _scene(tmp_path)
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    dataset = prepare_grut_dataset(cfg, manifest)
    args = grut_train_args(
        cfg,
        dataset,
        output_dir=tmp_path / "out",
        output_usd=tmp_path / "out" / "splat.usd",
    )
    assert "export_usd.format=standard" in args
    assert "export_usd.sorting_mode_hint=rayHitDistance" in args
    assert "dataset.test_split_interval=4" in args


def test_grut_args_include_overrides_and_iterations(tmp_path):
    cfg = _scene(tmp_path)
    cfg.reconstruction.grut_max_iterations = 30000
    cfg.reconstruction.grut_overrides = [
        "scheduler.positions.max_steps=30000",
        "strategy.densify.end_iteration=15000",
        "",  # ignored
    ]
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )
    dataset = prepare_grut_dataset(cfg, manifest)
    args = grut_train_args(
        cfg,
        dataset,
        output_dir=tmp_path / "out",
        output_usd=tmp_path / "out" / "splat.usd",
    )
    assert "n_iterations=30000" in args
    assert "scheduler.positions.max_steps=30000" in args
    assert "strategy.densify.end_iteration=15000" in args


def test_reconstruction_config_loads_grut_overrides():
    from scan2usd.config import ReconstructionConfig

    cfg = ReconstructionConfig.from_dict(
        {
            "grut_max_iterations": 30000,
            "grut_overrides": [
                "scheduler.positions.max_steps=30000",
                "loss.lambda_ssim=0.25",
            ],
        }
    )
    assert cfg.grut_max_iterations == 30000
    assert cfg.grut_overrides == [
        "scheduler.positions.max_steps=30000",
        "loss.lambda_ssim=0.25",
    ]


def test_grut_args_use_configured_usd_splat_format(tmp_path):
    """NuRec is the default export; 'standard' stays available as an override."""
    from scan2usd.config import SceneConfig
    from scan2usd.reconstruction.grut import GrutDataset, grut_train_args

    dataset = GrutDataset(
        root=tmp_path,
        images_dir=tmp_path / "images",
        sparse_dir=tmp_path / "sparse",
        held_out_manifest=tmp_path / "held_out.json",
        test_split_interval=10,
        masked_pixels_fraction=0.0,
    )
    cfg = SceneConfig()
    args = grut_train_args(cfg, dataset, output_dir=tmp_path, output_usd=tmp_path / "s.usd")
    # standard exposes per-Gaussian positions, which cleanup and the collision
    # mesh both need; nurec hides them in opaque field assets.
    assert "export_usd.format=standard" in args

    cfg.reconstruction.usd_splat_format = "nurec"
    args = grut_train_args(cfg, dataset, output_dir=tmp_path, output_usd=tmp_path / "s.usdz")
    assert "export_usd.format=nurec" in args


def test_downscaled_sparse_rescales_intrinsics(tmp_path, monkeypatch):
    """Staged images and COLMAP intrinsics must be downscaled together."""
    from scan2usd.config import SceneConfig
    from scan2usd.reconstruction import grut

    txt = tmp_path / "convert" / "txt"
    txt.mkdir(parents=True)
    (txt / "cameras.txt").write_text(
        "# comment\n1 OPENCV 3840 2160 1000 1000 1920 1080 0.1 0.2 0.3 0.4\n"
    )
    captured = {}

    class FakeAdapter:
        def __init__(self, *a, **k): pass
        def run(self, *args, **kwargs): captured.setdefault("calls", []).append(args)

    monkeypatch.setattr(grut, "ExternalToolAdapter", FakeAdapter)
    monkeypatch.setattr(grut, "resolve_colmap", lambda cfg: "colmap")
    monkeypatch.setattr(grut.shutil, "rmtree", lambda *a, **k: None)

    src = tmp_path / "sparse"
    (src / "0").mkdir(parents=True)
    target = tmp_path / "out"
    # Pre-create the txt the converter would have written.
    monkeypatch.setattr(
        grut.Path, "is_file", lambda self: self.name == "cameras.txt" or Path.is_file(self)
    )
    (target / "_colmap_convert" / "txt").mkdir(parents=True)
    (target / "_colmap_convert" / "txt" / "cameras.txt").write_text(
        "1 OPENCV 3840 2160 1000 1000 1920 1080 0.1 0.2 0.3 0.4\n"
    )
    grut._materialize_grut_sparse(SceneConfig(), src, target, downscale=2)
    line = (target / "_colmap_convert" / "txt" / "cameras.txt").read_text().split("\n")[0].split()
    assert line[1] == "PINHOLE"
    assert line[2:4] == ["1920", "1080"]          # image size halved
    assert [float(v) for v in line[4:8]] == [500.0, 500.0, 960.0, 540.0]  # fx fy cx cy halved


def test_export_recovers_from_checkpoint_when_inline_export_ooms(tmp_path, monkeypatch):
    """Training finishing but the in-process export dying must not lose the run."""
    from scan2usd.config import SceneConfig
    from scan2usd.pipeline.manifest import SceneManifest
    from scan2usd.reconstruction import grut

    build_root = tmp_path / "workspace" / "build" / "visual"
    (build_root / "run").mkdir(parents=True)
    (build_root / "run" / "ckpt_last.pt").write_bytes(b"ckpt")
    output = build_root / "environment_splat.usd"

    calls = []

    class FakeAdapter:
        def run(self, *args, **kwargs):
            calls.append(args)
            if args and args[0] == "train.py":
                raise RuntimeError("CUDA out of memory")
            output.write_text("#usda 1.0\n")  # the recovery export succeeds

    monkeypatch.setattr(grut, "resolve_grut", lambda cfg: (FakeAdapter(), tmp_path / "train.py"))
    monkeypatch.setattr(
        grut, "prepare_grut_dataset",
        lambda cfg, m, **k: grut.GrutDataset(
            root=tmp_path, images_dir=tmp_path, sparse_dir=tmp_path,
            held_out_manifest=tmp_path / "h.json", test_split_interval=10,
            masked_pixels_fraction=0.0,
        ),
    )
    cfg = SceneConfig()
    cfg.workspace_dir = tmp_path / "workspace"
    cfg.reconstruction.splat_cleanup.enabled = False
    manifest = SceneManifest.create(
        scene_name="r", source_config=tmp_path / "s.yaml", build_mode="preview"
    )
    result = grut.export_environment_particlefield(cfg, manifest)

    assert result.is_file()
    assert any("export_usd" in str(a) for call in calls for a in call)
    assert any("recovered by re-exporting" in w for w in manifest.warnings)


def test_pruning_ends_where_densification_ends(tmp_path):
    """
    The bug that left 65% of the bedroom's Gaussians dead.

    3DGRUT ends pruning at a hardcoded 15000 while opacity resets follow
    densification, so stretching only densification lets resets kill Gaussians
    for thousands of iterations after the last prune that could clear them.
    """
    from scan2usd.reconstruction.grut import consistent_strategy_overrides

    overrides = dict(o.split("=", 1) for o in consistent_strategy_overrides(50000, 0.5))
    assert overrides["strategy.densify.end_iteration"] == "25000"
    assert (
        overrides["strategy.prune.end_iteration"]
        == overrides["strategy.densify.end_iteration"]
    )
    assert overrides["scheduler.positions.max_steps"] == "50000"


def test_user_overrides_win_over_generated_ones(tmp_path):
    cfg = _scene(tmp_path)
    cfg.reconstruction.grut_max_iterations = 50000
    cfg.reconstruction.grut_overrides = ["strategy.prune.end_iteration=9000"]
    manifest = SceneManifest.create(
        scene_name="room", source_config=tmp_path / "s.yaml", build_mode="preview"
    )
    args = grut_train_args(
        cfg,
        prepare_grut_dataset(cfg, manifest),
        output_dir=tmp_path / "o",
        output_usd=tmp_path / "o" / "s.usd",
    )
    assert "strategy.prune.end_iteration=9000" in args
    assert "strategy.prune.end_iteration=25000" not in args
    # The rest of the generated schedule still applies.
    assert "strategy.densify.end_iteration=25000" in args


def test_anti_fog_is_off_until_asked_for(tmp_path):
    cfg = _scene(tmp_path)
    manifest = SceneManifest.create(
        scene_name="room", source_config=tmp_path / "s.yaml", build_mode="preview"
    )
    dataset = prepare_grut_dataset(cfg, manifest)
    args = grut_train_args(cfg, dataset, output_dir=tmp_path / "o", output_usd=tmp_path / "o" / "s.usd")
    assert not any("lambda_opacity" in a for a in args)

    cfg.reconstruction.anti_fog.enabled = True
    cfg.reconstruction.anti_fog.prune_weight_threshold = 0.5
    cfg.reconstruction.grut_max_iterations = 50000
    args = grut_train_args(cfg, dataset, output_dir=tmp_path / "o", output_usd=tmp_path / "o" / "s.usd")
    assert "loss.use_opacity=true" in args
    assert any(a.startswith("loss.lambda_opacity=") for a in args)
    # Weight pruning must not outlive densification, or it hollows the model out.
    assert "strategy.prune_weight.end_iteration=25000" in args
    assert "strategy.prune_weight.start_iteration=10000" in args


def _staged_bytes(images_dir):
    return {
        path.name: path.read_bytes()
        for path in sorted(images_dir.iterdir())
        if path.is_file()
    }


def test_parallel_staging_matches_the_serial_result_byte_for_byte(tmp_path, monkeypatch):
    """
    Correctness gate on the parallel staging path.

    Running the resizes across cores is only a speed change if the pixels are
    identical; anything else is a silent quality regression on every frame, and
    the training run would still complete and look plausible.
    """
    from scan2usd.reconstruction import grut

    serial = _scene(tmp_path / "serial")
    parallel = _scene(tmp_path / "parallel")
    for cfg in (serial, parallel):
        cfg.reconstruction.grut_downscale = 2

    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )

    original_stage_frames = grut._stage_frames

    def _forced_serial(jobs, *, index_path):
        results = [grut._stage_one(job) for job in jobs]
        return (
            sum(total for total, _masked in results),
            sum(masked for _total, masked in results),
        )

    monkeypatch.setattr(grut, "_stage_frames", _forced_serial)
    serial_dataset = prepare_grut_dataset(serial, manifest)
    serial_files = _staged_bytes(serial_dataset.images_dir)

    monkeypatch.setattr(grut, "_stage_frames", original_stage_frames)
    parallel_dataset = prepare_grut_dataset(parallel, manifest)

    assert _staged_bytes(parallel_dataset.images_dir) == serial_files
    assert parallel_dataset.masked_pixels_fraction == serial_dataset.masked_pixels_fraction


def test_staging_skips_unchanged_frames_and_restages_changed_ones(tmp_path, monkeypatch):
    """
    Staging used to redo every resize on every training run.

    That is the difference between a retrain paying for staging once and paying
    for it each time, so the skip has to be exercised — and it has to notice a
    frame that really did change, or a re-extraction would train on stale
    pixels while reporting success.
    """
    from scan2usd.reconstruction import grut

    cfg = _scene(tmp_path)
    cfg.reconstruction.grut_downscale = 2
    manifest = SceneManifest.create(
        scene_name="room",
        source_config=tmp_path / "scene.yaml",
        build_mode="preview",
    )

    staged: list[str] = []
    real_stage_one = grut._stage_one

    def _counting(job):
        staged.append(job.image_path.name)
        return real_stage_one(job)

    monkeypatch.setattr(grut, "_stage_one", _counting)
    # The pool would run the unpatched function in its workers, so force the
    # in-process path; parallel-vs-serial equivalence is covered above.
    monkeypatch.setattr(grut.os, "cpu_count", lambda: 1)

    first = prepare_grut_dataset(cfg, manifest)
    assert len(staged) == 4

    staged.clear()
    second = prepare_grut_dataset(cfg, manifest)
    assert staged == [], "re-staged frames that had not changed"
    assert second.masked_pixels_fraction == first.masked_pixels_fraction

    staged.clear()
    changed = cfg.nerfstudio_data_dir / "images" / "frame_001.jpg"
    Image.new("RGB", (20, 10), color=(7, 7, 7)).save(changed)
    prepare_grut_dataset(cfg, manifest)
    assert staged == ["frame_001.jpg"], "missed a frame whose source changed"


def test_restaging_at_a_new_downscale_never_writes_through_a_symlink(tmp_path):
    """
    Regression: switching ``grut_downscale`` 1 → 2 overwrote the source capture.

    At downscale 1 staging symlinks each staged frame back to
    ``ns_data/images``. At downscale 2 it writes a resized JPEG to the same
    path — and ``PIL.Image.save`` follows a symlink, so the resized copy landed
    on top of the original. On the bedroom scene that destroyed all 928
    full-resolution frames, silently, while the run reported success.

    The originals are the one artifact the pipeline cannot regenerate, so the
    check is on the source bytes, not on the staged output.
    """
    from scan2usd.reconstruction.grut import _stage_image

    source = tmp_path / "ns_data" / "images" / "frame_00001.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (64, 32), color=(200, 100, 50)).save(source)
    original = source.read_bytes()

    staged = tmp_path / "staged" / "frame_00001.jpg"
    _stage_image(source, staged, 1)
    assert staged.is_symlink(), "downscale 1 is expected to symlink; the hazard needs one"

    _stage_image(source, staged, 2)
    assert source.read_bytes() == original, "staging overwrote the source capture"
    assert not staged.is_symlink()
    with Image.open(staged) as restaged:
        assert restaged.size == (32, 16)


def test_failed_training_never_recovers_a_previous_run(tmp_path, monkeypatch):
    """
    Regression: a crashed run re-exported the *previous* run's checkpoint.

    The recovery path exists for a real failure — 3DGRUT exports from inside
    the training process and can run out of VRAM after training succeeded — but
    it caught every exception and searched all of ``build/visual`` by mtime. So
    ``optimizer.type=selective_adam``, whose CUDA plugin was not compiled,
    raised on startup and the pipeline exported last night's plain-Adam model,
    reported success, and produced a preview that looked entirely correct.

    Nothing downstream can detect that: the USD is valid, the cleanup report is
    consistent, the metrics are good. The only defence is refusing to recover
    from a checkpoint this run did not write.
    """
    import time

    from scan2usd.reconstruction.grut import find_latest_grut_checkpoint

    build_root = tmp_path / "visual"
    stale = build_root / "bedroom_static" / "run-0808_234359"
    stale.mkdir(parents=True)
    (stale / "ckpt_last.pt").write_bytes(b"previous run")

    started_at = time.time() + 1.0  # this run begins after the stale checkpoint
    assert find_latest_grut_checkpoint(build_root) is not None, "unbounded search finds it"
    assert find_latest_grut_checkpoint(build_root, newer_than=started_at) is None, (
        "recovery accepted a checkpoint from an earlier run"
    )

    fresh = build_root / "bedroom_static" / "run-0809_011600"
    fresh.mkdir(parents=True)
    checkpoint = fresh / "ckpt_last.pt"
    checkpoint.write_bytes(b"this run")
    import os

    os.utime(checkpoint, (started_at + 1, started_at + 1))
    assert find_latest_grut_checkpoint(build_root, newer_than=started_at) == checkpoint
