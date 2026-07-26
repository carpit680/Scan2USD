"""Mesh cleanup, canonicalization, simplification, and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scan2usd.geometry.frames import as_transform


@dataclass(frozen=True)
class MeshReport:
    path: str
    vertices: int
    faces: int
    watertight: bool
    volume_m3: float | None
    bounds_min: list[float]
    bounds_max: list[float]
    extents_m: list[float]
    components: int


def _trimesh():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Mesh processing requires the geometry extra: "
            'pip install -e ".[geometry]"'
        ) from exc
    return trimesh


def load_mesh(path: Path):
    trimesh = _trimesh()
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No triangle geometry in {path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh payload in {path}: {type(loaded).__name__}")
    return loaded


def clean_mesh(mesh, *, fill_small_holes: bool = False):
    mesh = mesh.copy()
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    if fill_small_holes:
        try:
            mesh.fill_holes()
        except (ValueError, RuntimeError):
            pass
    return mesh


def simplify_mesh(mesh, target_faces: int):
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        if simplified is not None and len(simplified.faces):
            return clean_mesh(simplified)
    except (ImportError, ValueError, RuntimeError, AttributeError):
        pass
    # Deterministic fallback: preserve every Nth face and clean unused vertices.
    step = max(1, int(np.ceil(len(mesh.faces) / target_faces)))
    fallback = mesh.copy()
    fallback.update_faces(np.arange(0, len(mesh.faces), step, dtype=np.int64))
    fallback.remove_unreferenced_vertices()
    return clean_mesh(fallback)


def transform_mesh(mesh, matrix: np.ndarray | list[list[float]]):
    transformed = mesh.copy()
    transformed.apply_transform(as_transform(matrix))
    return transformed


def mesh_report(mesh, path: Path) -> MeshReport:
    components = mesh.split(only_watertight=False)
    volume = float(abs(mesh.volume)) if mesh.is_watertight else None
    return MeshReport(
        path=str(path.resolve()),
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        volume_m3=volume,
        bounds_min=[float(v) for v in mesh.bounds[0]],
        bounds_max=[float(v) for v in mesh.bounds[1]],
        extents_m=[float(v) for v in mesh.extents],
        components=len(components),
    )


def process_mesh_file(
    source: Path,
    target: Path,
    *,
    source_to_usd: np.ndarray | None = None,
    target_faces: int | None = None,
    fill_small_holes: bool = False,
    report_path: Path | None = None,
) -> MeshReport:
    mesh = clean_mesh(load_mesh(source), fill_small_holes=fill_small_holes)
    if source_to_usd is not None:
        mesh = transform_mesh(mesh, source_to_usd)
    if target_faces is not None:
        mesh = simplify_mesh(mesh, target_faces)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(target)
    report = mesh_report(mesh, target)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return report


def split_static_mesh(mesh, output_dir: Path) -> dict[str, Path]:
    """Split connected components into floor, architecture, and obstacle debug chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    components = list(mesh.split(only_watertight=False))
    if not components:
        components = [mesh]
    z_min = float(mesh.bounds[0, 2])
    groups: dict[str, list] = {"floor": [], "architecture": [], "obstacles": []}
    for component in components:
        mean_normal = np.mean(component.face_normals, axis=0) if len(component.faces) else np.zeros(3)
        low = float(component.bounds[0, 2]) <= z_min + 0.15
        horizontal = abs(float(mean_normal[2])) >= 0.75
        if low and horizontal:
            groups["floor"].append(component)
        elif float(component.area) >= max(1.0, float(mesh.area) * 0.05):
            groups["architecture"].append(component)
        else:
            groups["obstacles"].append(component)
    trimesh = _trimesh()
    result: dict[str, Path] = {}
    for name, members in groups.items():
        if not members:
            continue
        combined = clean_mesh(trimesh.util.concatenate(members))
        path = output_dir / f"{name}.ply"
        combined.export(path)
        result[name] = path
    return result
