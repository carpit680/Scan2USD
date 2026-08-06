"""Resumable orchestration for the hybrid Scan-to-USD build graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

from scan2usd.assets.materials import build_object_materials
from scan2usd.assets.object_builder import reconstruct_object
from scan2usd.config import SceneConfig
from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD
from scan2usd.geometry.static_scene import build_static_scene
from scan2usd.lighting.estimate import estimate_scene_lighting
from scan2usd.pipeline.manifest import (
    CaptureRecord,
    ScaleEvidence,
    SceneManifest,
    TransformRecord,
)
from scan2usd.reconstruction.grut import export_environment_particlefield
from scan2usd.segmentation.propagate import propagate_with_sam2
from scan2usd.segmentation.propose import propose_with_yolo_world, save_proposals
from scan2usd.usd.package import build_usd_package


class ReviewRequired(RuntimeError):
    pass


T = TypeVar("T")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineOrchestrator:
    def __init__(
        self,
        cfg: SceneConfig,
        config_path: Path,
        *,
        build_mode: str = "production",
    ) -> None:
        self.cfg = cfg
        self.config_path = config_path.resolve()
        self.build_root = cfg.workspace_dir / "build"
        self.manifest_path = cfg.workspace_dir / "scene_manifest.json"
        self.state_path = self.build_root / "pipeline_state.json"
        self.build_root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            self.manifest = SceneManifest.load(self.manifest_path)
            if self.manifest.build_mode != build_mode:
                raise RuntimeError(
                    f"Existing manifest is {self.manifest.build_mode}; requested {build_mode}. "
                    "Use a separate workspace or remove the manifest intentionally."
                )
        else:
            self.manifest = self._create_manifest(build_mode)
            self.manifest.save(self.manifest_path)
        self.state = self._load_state()

    def _create_manifest(self, build_mode: str) -> SceneManifest:
        manifest = SceneManifest.create(
            scene_name=self.cfg.name,
            source_config=self.config_path,
            build_mode=build_mode,
        )
        source_path = self.cfg.video_path or self.cfg.frames_dir
        manifest.captures.append(
            CaptureRecord(
                capture_id="scene",
                kind="scene",
                path=str(Path(source_path).resolve()),
                modality=self.cfg.capture.modality,
                calibration_path=(
                    str(self.cfg.capture.calibration_path.resolve())
                    if self.cfg.capture.calibration_path
                    else None
                ),
                registered=True,
            )
        )
        if self.cfg.capture.modality in {"rgbd", "lidar"}:
            manifest.scale = ScaleEvidence(
                method="metric_sensor",
                meters_per_source_unit=1.0,
                confidence=0.95,
                reference=(
                    str(self.cfg.capture.calibration_path)
                    if self.cfg.capture.calibration_path
                    else self.cfg.capture.modality
                ),
                approved=True,
            )
        elif self.cfg.capture.scale_anchor_m is not None:
            manifest.scale = ScaleEvidence(
                method="known_length_pending_measurement",
                confidence=0.5,
                reference=f"{self.cfg.capture.scale_anchor_m} m anchor",
                approved=False,
            )
        calibration = self.cfg.capture.calibration_path
        if calibration and calibration.suffix.lower() == ".json" and calibration.is_file():
            raw = json.loads(calibration.read_text(encoding="utf-8"))
            matrix = raw.get("colmap_to_usd")
            if matrix is not None:
                manifest.transforms.append(
                    TransformRecord(
                        source_frame=FRAME_COLMAP,
                        target_frame=FRAME_USD,
                        matrix=matrix,
                        confidence=float(raw.get("registration_confidence", 1.0)),
                        evidence=str(calibration.resolve()),
                    )
                )
                meters = float(raw.get("meters_per_source_unit", 1.0))
                manifest.scale = ScaleEvidence(
                    method="calibrated_registration",
                    meters_per_source_unit=meters,
                    confidence=float(raw.get("registration_confidence", 1.0)),
                    reference=str(calibration.resolve()),
                    approved=bool(raw.get("approved", True)),
                )
        if not manifest.transforms:
            manifest.warnings.append(
                "No COLMAP→USD metric transform. Production export will remain blocked "
                "until calibration/review supplies one."
            )
        if self.cfg.capture.clean_plate_dir:
            manifest.captures.append(
                CaptureRecord(
                    capture_id="clean_plate",
                    kind="clean_plate",
                    path=str(self.cfg.capture.clean_plate_dir.resolve()),
                    modality=self.cfg.capture.modality,
                    registered=False,
                )
            )
        return manifest

    def _load_state(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"schema_version": "1.0", "stages": {}}

    def _save(self) -> None:
        self.manifest.save(self.manifest_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def _artifact_ready(self, artifact_id: str) -> bool:
        artifact = self.manifest.artifact(artifact_id)
        return bool(artifact and Path(artifact.path).is_file())

    def _baked_ready(self, artifact_id: str) -> bool:
        """
        Ready check for artifacts baked under the COLMAP→USD transform.

        Returns False when the manifest transform changed after the bake (or the
        artifact predates transform-hash stamping), forcing a coherent re-bake.
        """
        artifact = self.manifest.artifact(artifact_id)
        if not (artifact and Path(artifact.path).is_file()):
            return False
        current = self.manifest.colmap_to_usd_hash()
        if current is None:
            return True
        return artifact.metadata.get("transform_hash") == current

    def _instance_masks_ready(self) -> bool:
        """True only when the mask report exists and on-disk mask PNGs are present."""
        if not self._artifact_ready("instance_masks"):
            return False
        masks_root = Path(
            self.cfg.segmentation.masks_dir or (self.cfg.workspace_dir / "masks")
        )
        if not masks_root.is_dir():
            return False
        for path in masks_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                return True
        return False

    def run_stage(
        self,
        name: str,
        operation: Callable[[], T],
        *,
        force: bool = False,
        ready: Callable[[], bool] | None = None,
    ) -> T | None:
        stage = self.state["stages"].get(name, {})
        if not force and stage.get("status") == "completed" and (ready is None or ready()):
            return None
        self.state["stages"][name] = {"status": "running", "started_at": _now()}
        self._save()
        try:
            result = operation()
        except Exception as exc:
            self.state["stages"][name] = {
                "status": "failed",
                "finished_at": _now(),
                "error": str(exc),
            }
            self._save()
            raise
        self.state["stages"][name] = {"status": "completed", "finished_at": _now()}
        self._save()
        return result

    def align_floor(
        self,
        *,
        force: bool = False,
        distance_thresh: float = 0.04,
    ) -> Path:
        """Estimate floor plane and store COLMAP→USD rigid alignment (scale=1)."""
        from scan2usd.geometry.floor_align import estimate_floor_alignment, write_alignment_json
        from scan2usd.geometry.frames import uniform_scale

        sparse = self.cfg.nerfstudio_data_dir / "colmap" / "sparse" / "0"
        if not sparse.is_dir():
            raise RuntimeError(f"Missing COLMAP sparse model: {sparse}")

        def _has_floor_transform() -> bool:
            for item in self.manifest.transforms:
                if item.source_frame != FRAME_COLMAP or item.target_frame != FRAME_USD:
                    continue
                evidence = (item.evidence or "").lower()
                if "floor" in evidence or item.confidence > 0:
                    return Path(item.evidence or "").is_file() if item.evidence else True
            return False

        out = self.cfg.workspace_dir / "colmap_to_usd_floor.json"

        def estimate() -> Path:
            alignment = estimate_floor_alignment(
                sparse,
                distance_thresh=distance_thresh,
                min_inlier_ratio=self.cfg.geometry.min_floor_inlier_ratio,
                max_points_below=self.cfg.geometry.max_points_below_floor,
            )
            write_alignment_json(alignment, out)
            value = alignment.colmap_to_usd
            self.manifest.transforms = [
                item
                for item in self.manifest.transforms
                if not (item.source_frame == FRAME_COLMAP and item.target_frame == FRAME_USD)
            ]
            self.manifest.transforms.append(
                TransformRecord(
                    source_frame=FRAME_COLMAP,
                    target_frame=FRAME_USD,
                    matrix=value.tolist(),
                    confidence=float(alignment.floor.inlier_ratio),
                    evidence=str(out.resolve()),
                )
            )
            self.manifest.scale = ScaleEvidence(
                method="floor_plane_alignment_unit_scale",
                meters_per_source_unit=uniform_scale(value),
                confidence=float(alignment.floor.inlier_ratio),
                reference=str(out.resolve()),
                approved=self.manifest.build_mode == "preview",
            )
            self.manifest.approve(
                "metric_transform",
                reviewer="floor-align",
                notes="floor-plane RANSAC alignment (Z-up, floor at Z=0)",
            )
            self.manifest.register_artifact(
                artifact_id="floor_alignment",
                kind="colmap_to_usd_transform",
                path=out,
                producer="floor_plane_ransac",
                metadata={
                    "inliers": alignment.floor.inlier_count,
                    "inlier_ratio": alignment.floor.inlier_ratio,
                    "point_count": alignment.point_count,
                },
            )
            return out

        result = self.run_stage(
            "floor_alignment",
            estimate,
            force=force,
            ready=lambda: _has_floor_transform() and out.is_file(),
        )
        self._save()
        return result if result is not None else out

    def cleanup_splat(self, *, force: bool = False) -> Path:
        """Remove stray Gaussians from the environment ParticleField (no retrain)."""
        from scan2usd.reconstruction.splat_cleanup import (
            cleanup_particlefield,
            write_report_json,
        )

        artifact = self.manifest.artifact("environment_splat")
        if artifact is None or not Path(artifact.path).is_file():
            raise RuntimeError(
                "Missing environment_splat artifact; run build-visual-usd / reconstruct first"
            )
        splat_path = Path(artifact.path)
        build_root = splat_path.parent
        raw_backup = build_root / "environment_splat_raw.usd"
        report_path = build_root / "splat_cleanup_report.json"
        params = self.cfg.reconstruction.splat_cleanup.to_params()

        def run_cleanup() -> Path:
            # Prefer cleaning from the raw backup so threshold changes are re-runnable.
            source = raw_backup if raw_backup.is_file() else splat_path
            if source == splat_path and not raw_backup.is_file():
                # First cleanup: seed raw backup from current export.
                report = cleanup_particlefield(
                    self.cfg,
                    splat_path,
                    splat_path,
                    params,
                    raw_backup_path=raw_backup,
                )
            else:
                report = cleanup_particlefield(
                    self.cfg,
                    source,
                    splat_path,
                    params,
                    raw_backup_path=raw_backup if not raw_backup.is_file() else None,
                )
            write_report_json(report, report_path)
            meta = dict(artifact.metadata or {})
            meta["splat_cleanup"] = report.to_dict()
            meta["splat_cleanup_report"] = str(report_path.resolve())
            meta["raw_backup"] = str(raw_backup.resolve())
            self.manifest.register_artifact(
                artifact_id="environment_splat",
                kind=artifact.kind,
                path=splat_path,
                producer=artifact.producer,
                metadata=meta,
            )
            return splat_path

        result = self.run_stage(
            "splat_cleanup",
            run_cleanup,
            force=force,
            ready=lambda: report_path.is_file() and splat_path.is_file(),
        )
        self._save()
        return result if result is not None else splat_path

    def segment(self, *, force: bool = False) -> None:
        proposals_path = self.build_root / "segmentation" / "proposals.json"

        def propose():
            proposals = propose_with_yolo_world(
                self.cfg,
                images_dir=self.cfg.nerfstudio_data_dir / "images",
                prompts=list(self.cfg.classes),
            )
            save_proposals(proposals, proposals_path)
            self.manifest.register_artifact(
                artifact_id="object_proposals",
                kind="open_vocabulary_proposals",
                path=proposals_path,
                producer="yolo-world",
                metadata={"instances": len(proposals)},
            )
            return proposals

        proposals = self.run_stage(
            "object_proposals",
            propose,
            force=force,
            ready=lambda: self._artifact_ready("object_proposals"),
        )
        if proposals is None:
            from scan2usd.segmentation.propose import load_proposals

            proposals = load_proposals(proposals_path)
        self.run_stage(
            "mask_propagation",
            lambda: propagate_with_sam2(self.cfg, self.manifest, proposals),
            force=force,
            ready=self._instance_masks_ready,
        )
        self._save()

    def build(self, *, force: bool = False) -> Path:
        # Floor alignment first: Z-up USD with floor at Z=0 (unit scale until metric anchor).
        self.align_floor(force=force)

        approved = [
            obj
            for obj in self.manifest.objects
            if obj.review_state == "approved"
        ]
        if not approved and not self.cfg.segmentation.allow_no_objects:
            if not self._instance_masks_ready():
                self.segment(force=force)
            raise ReviewRequired(
                "Mark keepers as approved in Review (status dropdown), then re-run build-usd: "
                f"scan2usd review {self.config_path}. For an environment-only build "
                "(everything stays in the splat), set segmentation.allow_no_objects: true."
            )

        self.run_stage(
            "visual_particlefield",
            lambda: export_environment_particlefield(self.cfg, self.manifest),
            force=force,
            ready=lambda: self._artifact_ready("environment_splat"),
        )
        if self.cfg.reconstruction.splat_cleanup.enabled:
            self.cleanup_splat(force=force)
        self.run_stage(
            "static_geometry",
            lambda: build_static_scene(
                self.cfg,
                self.manifest,
                manifest_path=self.manifest_path,
            ),
            force=force,
            ready=lambda: self._baked_ready("static_collision_mesh"),
        )
        for obj in self.manifest.objects:
            if not obj.movable or obj.review_state != "approved":
                continue
            self.run_stage(
                f"object_geometry:{obj.instance_id}",
                lambda obj_id=obj.instance_id: reconstruct_object(
                    self.cfg,
                    self.manifest,
                    obj_id,
                ),
                force=force,
                ready=lambda obj_id=obj.instance_id: self._baked_ready(
                    f"object_{obj_id}_collision"
                ),
            )
            current = self.manifest.get_object(obj.instance_id)
            self.run_stage(
                f"object_materials:{obj.instance_id}",
                lambda current=current: build_object_materials(
                    self.cfg,
                    self.manifest,
                    current,
                ),
                force=force,
                ready=lambda obj_id=obj.instance_id: self._artifact_ready(
                    f"object_{obj_id}_materials"
                ),
            )
        self.run_stage(
            "lighting",
            lambda: estimate_scene_lighting(self.cfg, self.manifest),
            force=force,
            ready=lambda: self._artifact_ready("scene_lighting"),
        )
        root = self.run_stage(
            "usd_package",
            lambda: build_usd_package(self.cfg, self.manifest),
            force=force,
            ready=lambda: self._baked_ready("root_usd"),
        )
        self.run_quality_gates()
        self._save()
        if root is None:
            root_artifact = self.manifest.artifact("root_usd")
            assert root_artifact is not None
            return Path(root_artifact.path)
        return root

    def run_quality_gates(self) -> dict:
        """
        Judge the artifacts this build produced, and say what to do about it.

        Runs at the end rather than per stage because every gate reads a report
        the stage already wrote, so nothing is recomputed and a build is never
        blocked by measurement. Failures land in the manifest warnings and in
        build/quality_gates.json, which the GUI's Quality page reads.

        Deliberately non-blocking: the point is that a bad artifact stops being
        *silent*, not that the build stops. A gate that halts a preview build
        would get switched off within a day.
        """
        import json as _json

        from scan2usd.eval.gates import (
            check_floor,
            check_fog,
            check_mesh_sanity,
            check_training_health,
            summarize,
        )

        def _read(path: Path) -> dict | None:
            try:
                return _json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            except (OSError, ValueError):
                return None

        build_dir = Path(self.cfg.workspace_dir) / "build"
        cleanup = _read(build_dir / "visual" / "splat_cleanup_report.json")
        mesh = _read(build_dir / "geometry" / "static_collision_report.json")
        floor_artifact = self.manifest.artifact("floor_alignment")
        floor = dict(floor_artifact.metadata) if floor_artifact else None

        observed = None
        if cleanup and (fog := cleanup.get("fog_metrics")):
            span = fog.get("room_span")
            if span:
                observed = [float(span) / 3.0] * 3  # rough per-axis scale of the room

        results = [
            check_training_health(cleanup),
            check_fog(cleanup),
            check_mesh_sanity(mesh, observed_extents=observed),
            check_floor(floor),
        ]
        payload = summarize(results)
        out = build_dir / "quality_gates.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        for result in results:
            if result.status in {"fail", "warn"}:
                message = f"[{result.status}] {result.name}: {result.summary}"
                if result.recommendation:
                    message += f" -> {result.recommendation}"
                if message not in self.manifest.warnings:
                    self.manifest.warnings.append(message)
        return payload
