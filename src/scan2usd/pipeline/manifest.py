"""Versioned, JSON-serializable scene manifest used by every production stage."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCENE_MANIFEST_VERSION = "1.0"
APPROVAL_STATES = {"pending", "approved", "rejected", "degraded"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScaleEvidence:
    method: str = "unknown"
    meters_per_source_unit: float | None = None
    confidence: float = 0.0
    reference: str | None = None
    approved: bool = False

    def is_metric(self) -> bool:
        return bool(
            self.approved
            and self.meters_per_source_unit is not None
            and self.meters_per_source_unit > 0
        )


@dataclass
class TransformRecord:
    source_frame: str
    target_frame: str
    matrix: list[list[float]]
    confidence: float = 1.0
    evidence: str | None = None

    def __post_init__(self) -> None:
        if len(self.matrix) != 4 or any(len(row) != 4 for row in self.matrix):
            raise ValueError("TransformRecord.matrix must be 4x4")


@dataclass
class CaptureRecord:
    capture_id: str
    kind: str
    path: str
    modality: str
    calibration_path: str | None = None
    registered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectRecord:
    instance_id: str
    display_name: str
    class_name: str = "unknown"
    movable: bool = True
    review_state: str = "pending"
    mask_dir: str | None = None
    source_capture_id: str = "scene"
    local_to_world: list[list[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observed_background_coverage: float = 0.0
    render_mesh: str | None = None
    collision_mesh: str | None = None
    baked_texture: str | None = None
    pbr_textures: dict[str, str] = field(default_factory=dict)
    physics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.review_state not in APPROVAL_STATES:
            raise ValueError(f"Unknown review state: {self.review_state}")


@dataclass
class ArtifactRecord:
    artifact_id: str
    kind: str
    path: str
    producer: str
    sha256: str | None = None
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneManifest:
    scene_name: str
    source_config: str
    schema_version: str = SCENE_MANIFEST_VERSION
    canonical_frame: str = "usd_world_z_up_meters"
    build_mode: str = "production"
    review_state: str = "pending"
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    scale: ScaleEvidence = field(default_factory=ScaleEvidence)
    transforms: list[TransformRecord] = field(default_factory=list)
    captures: list[CaptureRecord] = field(default_factory=list)
    objects: list[ObjectRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    tool_versions: dict[str, str] = field(default_factory=dict)
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != SCENE_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported manifest schema {self.schema_version}; "
                f"expected {SCENE_MANIFEST_VERSION}"
            )
        if self.build_mode not in {"preview", "production"}:
            raise ValueError("build_mode must be preview or production")
        if self.review_state not in APPROVAL_STATES:
            raise ValueError(f"Unknown review state: {self.review_state}")

    @classmethod
    def create(
        cls,
        *,
        scene_name: str,
        source_config: Path,
        build_mode: str = "production",
    ) -> SceneManifest:
        return cls(
            scene_name=scene_name,
            source_config=str(source_config.resolve()),
            build_mode=build_mode,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SceneManifest:
        data = dict(raw)
        data["scale"] = ScaleEvidence(**dict(data.get("scale") or {}))
        data["transforms"] = [TransformRecord(**item) for item in data.get("transforms", [])]
        data["captures"] = [CaptureRecord(**item) for item in data.get("captures", [])]
        data["objects"] = [ObjectRecord(**item) for item in data.get("objects", [])]
        data["artifacts"] = [ArtifactRecord(**item) for item in data.get("artifacts", [])]
        return cls(**data)

    @classmethod
    def load(cls, path: Path) -> SceneManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Manifest must be a JSON object: {path}")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now()
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    def get_object(self, instance_id: str) -> ObjectRecord:
        for obj in self.objects:
            if obj.instance_id == instance_id:
                return obj
        raise KeyError(instance_id)

    def upsert_object(self, obj: ObjectRecord) -> None:
        for index, current in enumerate(self.objects):
            if current.instance_id == obj.instance_id:
                self.objects[index] = obj
                return
        self.objects.append(obj)

    def register_artifact(
        self,
        *,
        artifact_id: str,
        kind: str,
        path: Path,
        producer: str,
        metadata: dict[str, Any] | None = None,
        hash_contents: bool = True,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=kind,
            path=str(path.resolve()),
            producer=producer,
            sha256=sha256_file(path) if hash_contents and path.is_file() else None,
            metadata=metadata or {},
        )
        self.artifacts = [a for a in self.artifacts if a.artifact_id != artifact_id]
        self.artifacts.append(record)
        return record

    def artifact(self, artifact_id: str) -> ArtifactRecord | None:
        return next((a for a in self.artifacts if a.artifact_id == artifact_id), None)

    def approve(self, gate: str, *, reviewer: str, notes: str = "") -> None:
        self.approvals[gate] = {
            "state": "approved",
            "reviewer": reviewer,
            "notes": notes,
            "timestamp": _utc_now(),
        }

    def require_gate(self, gate: str) -> None:
        entry = self.approvals.get(gate)
        if not entry or entry.get("state") != "approved":
            raise RuntimeError(f"Manifest gate {gate!r} has not been approved")

    def require_production_ready(self) -> None:
        if self.build_mode != "production":
            return
        if not self.scale.is_metric():
            raise RuntimeError("Production USD requires approved metric scale evidence")
        from scan2usd.geometry.frames import FRAME_COLMAP, FRAME_USD, TransformGraph

        graph = TransformGraph()
        for transform in self.transforms:
            graph.add(
                transform.source_frame,
                transform.target_frame,
                transform.matrix,
                evidence=transform.evidence,
                confidence=transform.confidence,
            )
        try:
            graph.resolve(FRAME_COLMAP, FRAME_USD)
        except KeyError:
            raise RuntimeError(
                "Production USD requires an approved COLMAP→USD metric transform"
            ) from None
        if not any(obj.review_state == "approved" for obj in self.objects):
            raise RuntimeError(
                "Production USD requires at least one object marked approved in Review"
            )
