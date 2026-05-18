"""Splatfacto / process_data YAML → CLI flags."""

from __future__ import annotations

from pathlib import Path

import yaml

from scan2usd.config import SceneConfig
from scan2usd.splatfacto_config import SplatfactoConfig


def test_splatfacto_yaml_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace" / "frames").mkdir(parents=True)
    cfg_path = tmp_path / "scene.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "frames_dir": "workspace/frames",
                "splatfacto": {
                    "use_bilateral_grid": True,
                    "cull_alpha_thresh": 0.005,
                    "background_color": "black",
                },
                "process_data": {"num_downscales": 2},
            }
        )
    )
    cfg = SceneConfig.load(cfg_path)
    assert cfg.splatfacto.use_bilateral_grid is True
    assert cfg.splatfacto.cull_alpha_thresh == 0.005
    assert cfg.process_data.num_downscales == 2


def test_ns_train_extra_args() -> None:
    sf = SplatfactoConfig(
        use_bilateral_grid=True,
        background_color="black",
        rasterize_mode="antialiased",
        cull_alpha_thresh=0.005,
        camera_res_scale_factor=1.0,
        model_num_downscales=1,
        camera_optimizer_mode="SO3xR3",
    )
    args = sf.ns_train_extra_args()
    assert "--pipeline.model.use-bilateral-grid" in args
    assert args[args.index("--pipeline.model.use-bilateral-grid") + 1] == "True"
    assert "--pipeline.datamanager.camera-res-scale-factor" in args
    assert "--pipeline.model.num-downscales" in args
    assert "--pipeline.model.camera-optimizer.mode" in args
