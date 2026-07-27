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


def test_grut_args_request_nurec_volume_by_default(tmp_path):
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
    assert "export_usd.format=nurec" in args
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
    assert "export_usd.format=nurec" in args

    cfg.reconstruction.usd_splat_format = "standard"
    args = grut_train_args(cfg, dataset, output_dir=tmp_path, output_usd=tmp_path / "s.usd")
    assert "export_usd.format=standard" in args


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
