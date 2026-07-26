"""Compose a metric uniform scale onto a COLMAP→USD floor alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scan2usd.geometry.frames import (
    as_transform,
    compose_similarity,
    uniform_scale,
    validate_similarity,
)


def load_colmap_to_usd_matrix(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    matrix = raw.get("colmap_to_usd") if isinstance(raw, dict) else raw
    return validate_similarity(matrix)


def meters_per_unit_from_lengths(*, known_length_m: float, source_length: float) -> float:
    if known_length_m <= 0 or source_length <= 0:
        raise ValueError("known_length_m and source_length must be positive")
    return float(known_length_m) / float(source_length)


def apply_uniform_metric_scale(
    colmap_to_usd: np.ndarray | list[list[float]],
    meters_per_source_unit: float,
) -> np.ndarray:
    """
    Scale a rigid/unit-scale COLMAP→USD transform into meters.

    ``p_m = s * (R p + t)`` with the same rotation as the floor alignment.
    """
    if meters_per_source_unit <= 0:
        raise ValueError("meters_per_source_unit must be positive")
    base = as_transform(colmap_to_usd)
    # Strip any existing uniform scale, then apply the requested metric scale.
    current = uniform_scale(base)
    rotation = base[:3, :3] / current
    translation = base[:3, 3] / current
    return validate_similarity(
        compose_similarity(
            rotation=rotation,
            translation=translation * float(meters_per_source_unit),
            scale=float(meters_per_source_unit),
        )
    )


def write_metric_transform_json(
    matrix: np.ndarray,
    path: Path,
    *,
    meters_per_source_unit: float,
    method: str,
    evidence: str,
    confidence: float = 0.9,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "colmap_to_usd": validate_similarity(matrix).tolist(),
        "meters_per_source_unit": float(meters_per_source_unit),
        "registration_confidence": float(confidence),
        "approved": True,
        "method": method,
        "evidence": evidence,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def resolve_floor_transform_path(workspace_dir: Path, manifest_transforms: list) -> Path:
    """Prefer the floor-alignment JSON; fall back to any COLMAP→USD evidence file."""
    preferred = workspace_dir / "colmap_to_usd_floor.json"
    if preferred.is_file():
        return preferred
    for item in manifest_transforms:
        evidence = getattr(item, "evidence", None) or (
            item.get("evidence") if isinstance(item, dict) else None
        )
        if evidence and Path(evidence).is_file():
            return Path(evidence)
    raise FileNotFoundError(
        f"No floor/COLMAP→USD transform JSON under {workspace_dir}; run align-floor first"
    )
