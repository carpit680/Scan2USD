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


def count_components(mesh) -> int:
    """
    Number of connected components, without materialising them.

    ``mesh.split()`` builds a full Trimesh per component, copying vertices and
    faces. On a fragmented Poisson surface that is tens of thousands of meshes
    and tens of GB of RAM — enough to get the process OOM-killed. Counting via
    face adjacency touches only index arrays.
    """
    trimesh = _trimesh()
    if not len(mesh.faces):
        return 0
    try:
        labels = trimesh.graph.connected_components(
            mesh.face_adjacency, node_count=len(mesh.faces)
        )
        return int(len(labels))
    except Exception:  # noqa: BLE001
        return 1


def mesh_report(mesh, path: Path) -> MeshReport:
    components_count = count_components(mesh)
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
        components=components_count,
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


def split_static_mesh(mesh, output_dir: Path, *, max_components: int = 2000) -> dict[str, Path]:
    """
    Split the mesh into floor, architecture, and obstacle debug chunks.

    Never calls ``mesh.split()``. That builds a full Trimesh per connected
    component — copying vertices, faces and caches — and on a real 278k-face room
    surface it consumed 27 GB and did not finish, taking the whole build down with
    the OOM killer. Components are grouped by face labels instead, so the only
    copies made are the three chunks actually written.
    """
    trimesh = _trimesh()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not len(mesh.faces):
        return {}

    try:
        labels = trimesh.graph.connected_components(
            mesh.face_adjacency, node_count=len(mesh.faces)
        )
    except Exception:  # noqa: BLE001
        labels = [np.arange(len(mesh.faces))]
    if len(labels) > max_components:
        print(
            f"[split_static_mesh] {len(labels):,} components exceeds {max_components:,}; "
            "skipping debug chunks.",
            flush=True,
        )
        return {}

    z_min = float(mesh.bounds[0, 2])
    total_area = float(mesh.area)
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    triangles_z = mesh.vertices[mesh.faces][:, :, 2]

    groups: dict[str, list[np.ndarray]] = {"floor": [], "architecture": [], "obstacles": []}
    for face_index in labels:
        face_index = np.asarray(face_index, dtype=np.int64)
        mean_normal_z = float(np.mean(face_normals[face_index, 2]))
        low = float(triangles_z[face_index].min()) <= z_min + 0.15
        area = float(face_areas[face_index].sum())
        if low and abs(mean_normal_z) >= 0.75:
            groups["floor"].append(face_index)
        elif area >= max(1.0, total_area * 0.05):
            groups["architecture"].append(face_index)
        else:
            groups["obstacles"].append(face_index)

    result: dict[str, Path] = {}
    for name, members in groups.items():
        if not members:
            continue
        chunk = mesh.copy()
        chunk.update_faces(np.concatenate(members))
        chunk.remove_unreferenced_vertices()
        path = output_dir / f"{name}.ply"
        clean_mesh(chunk).export(path)
        result[name] = path
    return result
