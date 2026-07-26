"""Canonical coordinate-frame and metric-scale utilities for USD production."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


FRAME_COLMAP = "colmap_world"
FRAME_NERFSTUDIO_SAVED = "nerfstudio_saved"
FRAME_SPLAT = "splat_training"
FRAME_USD = "usd_world_z_up_meters"


def as_transform(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3] = value
        value = out
    if value.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("Transform contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("Transform must have homogeneous last row [0, 0, 0, 1]")
    if abs(float(np.linalg.det(value[:3, :3]))) < 1e-12:
        raise ValueError("Transform is singular")
    return value


def uniform_scale(matrix: np.ndarray | list[list[float]]) -> float:
    linear = as_transform(matrix)[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    if not np.allclose(singular, singular.mean(), rtol=1e-5, atol=1e-8):
        raise ValueError("Production transforms must be rigid or uniform similarity transforms")
    return float(singular.mean())


def validate_similarity(matrix: np.ndarray | list[list[float]]) -> np.ndarray:
    value = as_transform(matrix)
    scale = uniform_scale(value)
    rotation = value[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Transform rotation is not orthonormal")
    if np.linalg.det(rotation) < 0:
        raise ValueError("Reflections are not allowed in the canonical transform graph")
    return value


def apply_transform(
    points: np.ndarray,
    matrix: np.ndarray | list[list[float]],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    original_shape = pts.shape
    if original_shape[-1:] != (3,):
        raise ValueError("Points must end in dimension 3")
    flat = pts.reshape(-1, 3)
    transform = as_transform(matrix)
    homogeneous = np.concatenate([flat, np.ones((len(flat), 1))], axis=1)
    return (homogeneous @ transform.T)[:, :3].reshape(original_shape)


def transform_pose(
    camera_to_source: np.ndarray,
    source_to_target: np.ndarray,
) -> np.ndarray:
    """Transform a camera-to-world pose between world frames."""
    return as_transform(source_to_target) @ as_transform(camera_to_source)


def compose_similarity(
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    out = np.eye(4, dtype=np.float64)
    rot = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3) or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-6):
        raise ValueError("rotation must be orthonormal 3x3")
    if np.linalg.det(rot) < 0:
        raise ValueError("rotation must be right-handed")
    out[:3, :3] = rot * float(scale)
    if translation is not None:
        out[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return out


@dataclass(frozen=True)
class TransformEdge:
    source: str
    target: str
    matrix: np.ndarray
    evidence: str | None = None
    confidence: float = 1.0


class TransformGraph:
    """Small bidirectional graph with one authoritative transform per frame pair."""

    def __init__(self) -> None:
        self._edges: dict[tuple[str, str], TransformEdge] = {}

    def add(
        self,
        source: str,
        target: str,
        matrix: np.ndarray | list[list[float]],
        *,
        evidence: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        if not source or not target or source == target:
            raise ValueError("Transform edge requires two distinct frame names")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        value = validate_similarity(matrix)
        reverse = np.linalg.inv(value)
        self._edges[(source, target)] = TransformEdge(
            source, target, value, evidence, confidence
        )
        self._edges[(target, source)] = TransformEdge(
            target, source, reverse, evidence, confidence
        )

    def resolve(self, source: str, target: str) -> np.ndarray:
        if source == target:
            return np.eye(4, dtype=np.float64)
        queue: deque[tuple[str, np.ndarray]] = deque([(source, np.eye(4))])
        visited = {source}
        while queue:
            current, source_to_current = queue.popleft()
            for (edge_source, edge_target), edge in self._edges.items():
                if edge_source != current or edge_target in visited:
                    continue
                source_to_next = edge.matrix @ source_to_current
                if edge_target == target:
                    return source_to_next
                visited.add(edge_target)
                queue.append((edge_target, source_to_next))
        raise KeyError(f"No transform path from {source!r} to {target!r}")

    def transform_points(self, points: np.ndarray, source: str, target: str) -> np.ndarray:
        return apply_transform(points, self.resolve(source, target))

    def records(self) -> list[TransformEdge]:
        """Return only one direction for each stored pair, deterministically."""
        unique: list[TransformEdge] = []
        seen: set[frozenset[str]] = set()
        for key in sorted(self._edges):
            edge = self._edges[key]
            pair = frozenset((edge.source, edge.target))
            if pair in seen:
                continue
            seen.add(pair)
            unique.append(edge)
        return unique
