"""Aggregate photorealism, cleanliness, and validation gates into one quality report.

``scene_quality.json`` is the single number the auto-tuner optimizes and the GUI
surfaces. Photorealism comes from Isaac renders at held-out capture cameras
(``tools/isaac/render_heldout.py``) compared against the real frames; cleanliness
from the splat-cleanup report; registration/gates from ``build_report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scan2usd.config import SceneConfig
from scan2usd.eval.photorealism import evaluate_held_out_renders


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def quality_score_breakdown(
    *,
    mean_psnr: float | None,
    mean_ssim: float | None,
    mean_lpips: float | None,
    registration_median_m: float | None,
    registration_threshold_m: float,
    failed_required_checks: int,
) -> dict[str, Any]:
    """
    Score plus its components, so a change in the total is always explainable.

    Penalties depend on ``build_report.json``, which only exists after
    ``validate-usd``. Comparing a tuner trial (no report yet) against a validated
    build therefore compares different totals — ``appearance_score`` is the part
    that is always comparable.
    """
    if mean_ssim is None and mean_psnr is None:
        return {
            "quality_score": None,
            "appearance_score": None,
            "registration_penalty": 0.0,
            "failed_check_penalty": 0.0,
        }
    ssim_term = _clamp(mean_ssim) if mean_ssim is not None else 0.0
    psnr_term = _clamp((mean_psnr or 0.0) / 40.0)
    if mean_lpips is not None:
        lpips_term = _clamp(1.0 - mean_lpips)
        appearance = 100.0 * (0.45 * ssim_term + 0.35 * lpips_term + 0.20 * psnr_term)
    else:
        appearance = 100.0 * (0.80 * ssim_term + 0.20 * psnr_term)
    registration_penalty = (
        10.0
        if registration_median_m is not None
        and registration_median_m > registration_threshold_m
        else 0.0
    )
    check_penalty = min(30.0, 5.0 * failed_required_checks)
    return {
        "quality_score": round(max(0.0, appearance - registration_penalty - check_penalty), 3),
        "appearance_score": round(appearance, 3),
        "registration_penalty": registration_penalty,
        "failed_check_penalty": check_penalty,
    }


def quality_score(
    *,
    mean_psnr: float | None,
    mean_ssim: float | None,
    mean_lpips: float | None,
    registration_median_m: float | None,
    registration_threshold_m: float,
    failed_required_checks: int,
) -> float | None:
    """
    Scalar 0–100 quality score (higher is better).

    Photorealism dominates: SSIM 45%, LPIPS 35% (weight folded into SSIM when
    LPIPS is unavailable), PSNR 20% (normalized at 40 dB). Penalties: −10 when
    splat/mesh registration exceeds the QA threshold, −5 per failed required
    validation gate (capped at −30). Returns None when no photorealism metrics
    exist — a scene with no measured appearance cannot be scored.
    """
    if mean_ssim is None and mean_psnr is None:
        return None
    ssim_term = _clamp(mean_ssim) if mean_ssim is not None else 0.0
    psnr_term = _clamp((mean_psnr or 0.0) / 40.0)
    if mean_lpips is not None:
        lpips_term = _clamp(1.0 - mean_lpips)
        score = 100.0 * (0.45 * ssim_term + 0.35 * lpips_term + 0.20 * psnr_term)
    else:
        score = 100.0 * (0.80 * ssim_term + 0.20 * psnr_term)
    if registration_median_m is not None and registration_median_m > registration_threshold_m:
        score -= 10.0
    score -= min(30.0, 5.0 * failed_required_checks)
    return round(max(0.0, score), 3)


def build_scene_quality_report(
    cfg: SceneConfig,
    *,
    render_dir: Path | None = None,
    compute_lpips: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Collect metrics from workspace artifacts and write ``scene_quality.json``."""
    usd_dir = Path(cfg.usd.output_dir or cfg.workspace_dir / "usd")
    grut_root = cfg.workspace_dir / "build" / "grut_dataset"
    heldout_spec = grut_root / "held_out.json"
    render_dir = render_dir or (cfg.workspace_dir / "build" / "heldout_renders")
    output_path = output_path or (usd_dir / "scene_quality.json")

    photorealism: dict[str, Any] | None = None
    if heldout_spec.is_file() and render_dir.is_dir():
        photorealism = evaluate_held_out_renders(
            heldout_spec,
            grut_root / "images",
            render_dir,
            output_path=usd_dir / "photorealism_report.json",
            compute_lpips=compute_lpips,
        )

    cleanup = _load_json(cfg.workspace_dir / "build" / "visual" / "splat_cleanup_report.json")
    cleanliness: dict[str, Any] = {}
    if cleanup:
        input_count = int(cleanup.get("input_count") or 0)
        if input_count > 0:
            cleanliness = {
                "gaussian_count": int(cleanup.get("kept_count") or 0),
                "removed_spatial_fraction": (cleanup.get("removed_spatial") or 0) / input_count,
                "removed_opacity_fraction": (cleanup.get("removed_opacity") or 0) / input_count,
                "outlier_std": cleanup.get("outlier_std"),
                "min_opacity": cleanup.get("min_opacity"),
            }

    build_report = _load_json(usd_dir / "build_report.json")
    registration_median: float | None = None
    failed_required: list[str] = []
    usable: bool | None = None
    if build_report:
        usable = bool(build_report.get("usable"))
        failed_required = list(build_report.get("failed_required_checks") or [])
        for check in build_report.get("checks", []):
            if check.get("name") == "splat_proxy_registration":
                registration_median = check.get("details", {}).get("median_vertex_distance_m")

    breakdown = quality_score_breakdown(
        mean_psnr=photorealism.get("mean_psnr") if photorealism else None,
        mean_ssim=photorealism.get("mean_ssim") if photorealism else None,
        mean_lpips=photorealism.get("mean_lpips") if photorealism else None,
        registration_median_m=registration_median,
        registration_threshold_m=cfg.qa.max_registration_error_m,
        failed_required_checks=len(failed_required),
    )
    report = {
        "schema_version": "1.0",
        "scene": cfg.name,
        "quality_score": breakdown["quality_score"],
        "score_breakdown": breakdown,
        "photorealism": {
            "mean_psnr": photorealism.get("mean_psnr") if photorealism else None,
            "mean_ssim": photorealism.get("mean_ssim") if photorealism else None,
            "mean_lpips": photorealism.get("mean_lpips") if photorealism else None,
            "evaluated_views": photorealism.get("evaluated") if photorealism else 0,
            "expected_views": photorealism.get("expected") if photorealism else 0,
            # Which resolution the comparison ran at. Scores from the two modes
            # are not interchangeable: on the same renders, switching from
            # "reference" to "render" left PSNR unmoved (18.87 -> 18.89) but
            # moved SSIM 0.798 -> 0.741, because upsampling the render used to
            # blur both images toward each other in exactly the flat regions
            # SSIM weights most. Any comparison across modes is meaningless.
            "eval_resolution": (
                photorealism.get("eval_resolution") if photorealism else None
            ),
            "render_dir": str(render_dir.resolve()) if render_dir.is_dir() else None,
            "note": (
                "Isaac RTX renders at held-out capture cameras vs real frames. "
                "Pinhole renders vs distorted captures bias edges slightly; the "
                "bias is constant across trials so relative comparison holds."
            ),
        },
        "cleanliness": cleanliness,
        "registration": {
            "median_vertex_distance_m": registration_median,
            "threshold_m": cfg.qa.max_registration_error_m,
        },
        "validation": {
            "usable": usable,
            "failed_required_checks": failed_required,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
