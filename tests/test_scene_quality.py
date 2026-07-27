"""scene_quality.json aggregation and scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.eval.scene_quality import build_scene_quality_report, quality_score


def _cfg(tmp_path: Path) -> SceneConfig:
    return SceneConfig._from_dict({"workspace_dir": str(tmp_path / "ws")}, base=tmp_path)


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def test_quality_score_weights_and_penalties() -> None:
    base = quality_score(
        mean_psnr=40.0,
        mean_ssim=1.0,
        mean_lpips=0.0,
        registration_median_m=0.01,
        registration_threshold_m=0.03,
        failed_required_checks=0,
    )
    assert base == 100.0
    penalized = quality_score(
        mean_psnr=40.0,
        mean_ssim=1.0,
        mean_lpips=0.0,
        registration_median_m=0.25,
        registration_threshold_m=0.03,
        failed_required_checks=2,
    )
    assert penalized == 100.0 - 10.0 - 10.0
    assert (
        quality_score(
            mean_psnr=None,
            mean_ssim=None,
            mean_lpips=None,
            registration_median_m=None,
            registration_threshold_m=0.03,
            failed_required_checks=0,
        )
        is None
    )


def test_build_report_aggregates_available_artifacts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    grut = cfg.workspace_dir / "build" / "grut_dataset"
    (grut).mkdir(parents=True)
    (grut / "held_out.json").write_text(
        json.dumps({"images": [{"file": "frame_0001.jpg", "camera_to_world": None}]})
    )
    _write_image(grut / "images" / "frame_0001.jpg", 128)
    renders = cfg.workspace_dir / "build" / "heldout_renders"
    _write_image(renders / "frame_0001.png", 128)

    visual = cfg.workspace_dir / "build" / "visual"
    visual.mkdir(parents=True)
    (visual / "splat_cleanup_report.json").write_text(
        json.dumps(
            {
                "input_count": 1000,
                "kept_count": 940,
                "removed_spatial": 50,
                "removed_opacity": 10,
                "removed_scale": 0,
                "outlier_std": 4.0,
                "min_opacity": 0.01,
            }
        )
    )
    usd_dir = Path(cfg.usd.output_dir)
    usd_dir.mkdir(parents=True)
    (usd_dir / "build_report.json").write_text(
        json.dumps(
            {
                "usable": False,
                "failed_required_checks": ["splat_proxy_registration"],
                "checks": [
                    {
                        "name": "splat_proxy_registration",
                        "details": {"median_vertex_distance_m": 0.2},
                    }
                ],
            }
        )
    )

    report = build_scene_quality_report(cfg, compute_lpips=False)
    assert (usd_dir / "scene_quality.json").is_file()
    photo = report["photorealism"]
    assert photo["evaluated_views"] == 1
    assert photo["mean_ssim"] is not None and photo["mean_ssim"] > 0.99
    assert report["cleanliness"]["gaussian_count"] == 940
    assert abs(report["cleanliness"]["removed_spatial_fraction"] - 0.05) < 1e-9
    assert report["registration"]["median_vertex_distance_m"] == 0.2
    assert report["validation"]["failed_required_checks"] == ["splat_proxy_registration"]
    # identical images → SSIM 1.0 but PSNR=inf is excluded from the mean (None),
    # so score = 80 (SSIM-only weight) − 10 registration − 5 one failed gate.
    assert report["quality_score"] == 65.0


def test_build_report_without_renders_scores_none(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    report = build_scene_quality_report(cfg, compute_lpips=False)
    assert report["quality_score"] is None
    assert report["photorealism"]["evaluated_views"] == 0


def test_score_breakdown_separates_appearance_from_penalties() -> None:
    """
    Appearance is the tuner-comparable part; penalties only exist post-validation.

    Regression guard for the confusing 74.1 -> 64.1 shift seen when validate-usd
    first populated a registration error for an unchanged scene.
    """
    from scan2usd.eval.scene_quality import quality_score_breakdown

    common = dict(mean_psnr=23.35, mean_ssim=0.8162, mean_lpips=0.2649)
    untouched = quality_score_breakdown(
        **common,
        registration_median_m=None,
        registration_threshold_m=0.03,
        failed_required_checks=0,
    )
    validated = quality_score_breakdown(
        **common,
        registration_median_m=12.6,
        registration_threshold_m=0.03,
        failed_required_checks=0,
    )
    # Same renders → identical appearance; only the penalty differs.
    assert untouched["appearance_score"] == validated["appearance_score"]
    assert validated["registration_penalty"] == 10.0
    assert (
        validated["quality_score"]
        == untouched["quality_score"] - validated["registration_penalty"]
    )
