"""SAM-style multi-view mask propagation and manifest import."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scan2usd.config import SceneConfig
from scan2usd.pipeline.manifest import ObjectRecord, SceneManifest
from scan2usd.reconstruction.external_cli import resolve_external_command
from scan2usd.segmentation.propose import InstanceProposal, save_proposals


def sam2_args(
    *,
    images_dir: Path,
    proposals_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "--images",
        str(images_dir.resolve()),
        "--proposals",
        str(proposals_path.resolve()),
        "--output",
        str(output_dir.resolve()),
        "--mask-format",
        "foreground-white",
    ]


def _mask_files(mask_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in mask_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file()
    )


def validate_instance_masks(
    cfg: SceneConfig,
    proposals: list[InstanceProposal],
    output_dir: Path,
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for proposal in proposals:
        mask_dir = output_dir / proposal.instance_id
        masks = _mask_files(mask_dir) if mask_dir.is_dir() else []
        valid = 0
        foreground_pixels = 0
        total_pixels = 0
        for path in masks:
            mask = np.asarray(Image.open(path).convert("L"))
            total_pixels += int(mask.size)
            foreground_pixels += int((mask > 127).sum())
            if np.any(mask > 127):
                valid += 1
        report[proposal.instance_id] = {
            "mask_dir": str(mask_dir.resolve()),
            "frames": len(masks),
            "valid_frames": valid,
            "foreground_fraction": foreground_pixels / max(total_pixels, 1),
            "meets_min_views": valid >= cfg.segmentation.min_views_per_object,
        }
    return report


def import_masks_into_manifest(
    cfg: SceneConfig,
    manifest: SceneManifest,
    proposals: list[InstanceProposal],
    output_dir: Path,
) -> dict[str, dict]:
    report = validate_instance_masks(cfg, proposals, output_dir)
    for proposal in proposals:
        item = report[proposal.instance_id]
        existing = next(
            (obj for obj in manifest.objects if obj.instance_id == proposal.instance_id),
            None,
        )
        obj = existing or ObjectRecord(
            instance_id=proposal.instance_id,
            display_name=proposal.instance_id.replace("_", " ").title(),
            class_name=proposal.class_name,
        )
        obj.class_name = proposal.class_name
        obj.mask_dir = item["mask_dir"]
        obj.review_state = "pending"
        if not item["meets_min_views"]:
            warning = (
                f"Only {item['valid_frames']} valid masks; "
                f"need {cfg.segmentation.min_views_per_object}"
            )
            if warning not in obj.warnings:
                obj.warnings.append(warning)
        manifest.upsert_object(obj)
    return report


def propagate_with_sam2(
    cfg: SceneConfig,
    manifest: SceneManifest,
    proposals: list[InstanceProposal],
    *,
    images_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Invoke a pinned SAM2 runner.

    Runner contract: accept ``sam2_args`` and write one foreground-white mask
    directory per proposal instance. This keeps SAM2's heavy environment isolated.
    """
    images_dir = images_dir or (cfg.nerfstudio_data_dir / "images")
    output_dir = output_dir or Path(cfg.segmentation.masks_dir or cfg.workspace_dir / "masks")
    job_dir = cfg.workspace_dir / "build" / "segmentation"
    proposals_path = save_proposals(proposals, job_dir / "proposals.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = resolve_external_command(
        cfg,
        "sam2_runner",
        default="",
        required=True,
    )
    assert prefix is not None
    from scan2usd.reconstruction.external_cli import ExternalToolAdapter

    adapter = ExternalToolAdapter("sam2", prefix)
    adapter.run(*sam2_args(images_dir=images_dir, proposals_path=proposals_path, output_dir=output_dir))
    report = import_masks_into_manifest(cfg, manifest, proposals, output_dir)
    report_path = job_dir / "mask_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest.register_artifact(
        artifact_id="instance_masks",
        kind="multiview_instance_masks",
        path=report_path,
        producer="sam2",
        metadata={"objects": len(proposals), "root": str(output_dir.resolve())},
    )
    return report_path
