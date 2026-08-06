"""Remove stray Gaussians from a ParticleField3DGaussianSplat USD."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SplatCleanupParams:
    """
    Thresholds for stray-Gaussian cleanup.

    ``outlier_std`` is a blunt global radius around the scene median: it cannot
    tell a floater from a distant wall, so it either leaves the floaters or eats
    real geometry. The three filters below are targeted instead, and together
    remove the Gaussians that obstruct exterior views while touching only a few
    percent of the model:

    - ``max_scale_frac`` — Gaussians bigger than this fraction of the scene
      diagonal. A handful of these blanket the whole view from outside.
    - ``crop_margin`` — drop everything beyond the observed volume (bounds of the
      SfM points) expanded by this fraction. This is the halo shell.
    - ``min_neighbors`` within ``neighbor_radius_frac`` — isolated Gaussians in
      free space, i.e. classic floaters near objects.
    """

    enabled: bool = True
    # See SplatCleanupConfig in config.py: 4.0 measurably over-prunes dense models.
    outlier_std: float = 8.0
    min_opacity: float = 0.01
    max_scale: float | None = None
    # MEASURED on the kitchen 50k model (interior PSNR vs the no-filter baseline
    # of 23.35 dB, exterior checked with tools/isaac/render_orbit.py):
    #   crop + giant-scale only ....... 22.62 dB  (-0.73)  far halo gone
    #   + needle + density ............ 17.49 dB  (-5.9)   exterior cleaner
    #   aggressive thresholds ......... 15.46 dB  (-7.9)   exterior cleanest
    # Removing the far halo and the handful of scene-sized Gaussians is nearly
    # free. Everything beyond that trades real interior fidelity for exterior
    # tidiness, because those Gaussians ARE doing work in the observed views.
    # Defaults therefore keep only the free part; the rest is opt-in.
    max_scale_frac: float | None = 0.08
    crop_margin: float | None = 0.5
    min_neighbors: int = 0
    neighbor_radius_frac: float = 0.01
    max_needle_ratio: float | None = None
    needle_min_length_frac: float = 0.005
    # Frustum visibility barely discriminates on inside-out room captures (median
    # Gaussian is seen by 208 of 787 cameras); useful for object-centric scans.
    min_view_count: int = 0
    # Refuse to emit a scene that cleanup has gutted. Removing nearly everything
    # means either the thresholds are wrong for this model or the model itself is
    # degenerate — an MCMC run whose relocate schedule ended early leaves ~99% of
    # Gaussians at zero opacity, and without this the build "succeeds" with an
    # empty scene. 0 disables the check.
    min_keep_fraction: float = 0.05
    # Free-space carving. Judges a Gaussian by where it sits against direct
    # camera evidence rather than by how it looks, which is what lets it reach
    # interior haze: opacity and scale thresholds cannot separate a hazy
    # Gaussian from a faint real one, but "the cameras saw straight through
    # here" is not a matter of degree. 0 disables.
    free_space_votes: int = 0
    carve_resolution: int = 256
    carve_max_rays: int = 400_000
    surface_radius_frac: float = 0.015
    # Independent of the carve: far from every SfM point AND locally sparse.
    # Reaches the haze rays cannot prove empty; a real but untracked surface
    # (a white ceiling) stays because it is still a densely packed sheet.
    air_min_neighbors: int = 0
    air_neighbor_radius_frac: float = 0.01
    # Also remove Gaussians the cameras saw straight past. Safe where the
    # density rule is not: anything that occludes is left alone.
    free_behind: bool = False
    # Remove all non-surface Gaussians inside the walked volume. The most
    # aggressive interior rule available; walls are outside the hull so it cannot
    # reach them, but poorly-tracked furniture can be.
    hull_air: bool = False


@dataclass
class SplatCleanupReport:
    input_count: int
    kept_count: int
    removed_spatial: int
    removed_opacity: int
    removed_scale: int
    outlier_std: float
    min_opacity: float
    max_scale: float | None
    input_path: str
    output_path: str
    raw_backup_path: str | None = None
    removed_crop: int = 0
    removed_density: int = 0
    removed_needle: int = 0
    removed_visibility: int = 0
    removed_free_space: int = 0
    free_space_breakdown: dict[str, Any] | None = None
    fog_metrics: dict[str, Any] | None = None
    free_space_error: str | None = None
    scene_diagonal: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_view_counts(
    points: np.ndarray,
    cameras: dict,
    images: dict,
    *,
    max_depth: float | None = None,
    margin_px: float = 0.0,
) -> np.ndarray:
    """
    Number of training cameras that actually see each Gaussian.

    This is the criterion the 3DGS literature uses to identify floaters
    (contribution-based trimming): geometry the capture genuinely observed is
    seen from many views, while halo shells and boundary spikes exist to satisfy
    one or two frames and appear in almost none. Unlike size or position
    thresholds it cannot mistake a large wall Gaussian for an artifact, which is
    what made the geometric filters trade away interior quality.

    Frustum visibility only — occlusion is not resolved, so a Gaussian hidden
    behind a wall still counts. That is deliberate: it keeps the test
    conservative, removing only what no camera could have seen.
    """
    from scan2usd.reconstruction.colmap_io import quat_to_rotmat

    pts = np.asarray(points, dtype=np.float64)
    counts = np.zeros(len(pts), dtype=np.int32)
    for pose in images.values():
        intrinsics = cameras.get(pose.camera_id)
        if intrinsics is None:
            continue
        params = np.asarray(intrinsics.params, dtype=np.float64)
        if intrinsics.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:  # PINHOLE, OPENCV and friends
            fx, fy = float(params[0]), float(params[1])
            cx, cy = float(params[2]), float(params[3])

        rotation = quat_to_rotmat(pose.qvec)
        camera_points = pts @ rotation.T + np.asarray(pose.tvec, dtype=np.float64)
        depth = camera_points[:, 2]
        visible = depth > 1e-6
        if max_depth is not None:
            visible &= depth < float(max_depth)
        if not visible.any():
            continue
        safe_depth = np.where(visible, depth, 1.0)
        u = fx * camera_points[:, 0] / safe_depth + cx
        v = fy * camera_points[:, 1] / safe_depth + cy
        visible &= (
            (u >= -margin_px)
            & (u < intrinsics.width + margin_px)
            & (v >= -margin_px)
            & (v < intrinsics.height + margin_px)
        )
        counts += visible
    return counts


def _neighbor_counts(points: np.ndarray, radius: float) -> np.ndarray:
    """Neighbours within ``radius`` per point (excluding self)."""
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        return np.asarray(tree.query_ball_point(points, radius, return_length=True)) - 1
    except ImportError:
        # Voxel-grid approximation: count occupants of the containing cell.
        keys = np.floor(points / max(radius, 1e-9)).astype(np.int64)
        _uniq, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True
        )
        return counts[inverse] - 1


def _carve(positions: np.ndarray, sparse_dir: Path, params: SplatCleanupParams):
    """Carve the volume the cameras saw through. Reused for removal and metrics."""
    from scan2usd.reconstruction.free_space import carve_from_colmap

    return carve_from_colmap(
        positions,
        sparse_dir,
        resolution=params.carve_resolution,
        max_rays=params.carve_max_rays,
    )


def fog_metrics_from_carve(
    carve: Any,
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    params: SplatCleanupParams,
) -> dict[str, Any]:
    """
    How hazy the air is, for the Gaussians given.

    Deliberately *not* the filter's own predicate. Re-running "carved free and
    not near a surface" on the set just filtered by that predicate returns zero
    every time by construction, which reports a perfectly clear room whatever
    the scene actually looks like. Haze is measured here by the independent
    half of the test — inside the camera path with no surface nearby — so
    Gaussians the carve could not prove empty still count against the score.

    ``transmittance_across_room`` is the number to watch: the fraction of a view
    that survives crossing the room, which is what "cloudy" means and what
    held-out PSNR cannot see.
    """
    from scan2usd.reconstruction.free_space import (
        inside_camera_hull,
        within_surface_radius,
    )

    positions = np.asarray(positions, dtype=np.float64)
    radius = float(params.surface_radius_frac) * float(
        np.linalg.norm(carve.reference[1] - carve.reference[0])
    )
    near_surface = within_surface_radius(positions, carve.points, radius)
    in_hull = inside_camera_hull(positions, carve.centres) & carve.grid.contains(positions)
    hull_fog = in_hull & ~near_surface

    scales_arr = np.asarray(scales, dtype=np.float64)
    scale_mid = np.sort(scales_arr, axis=1)[:, 1] if scales_arr.ndim == 2 else scales_arr
    cross_section = np.asarray(opacities, dtype=np.float64).reshape(-1) * np.pi * scale_mid**2

    from scan2usd.reconstruction.free_space import hull_voxel_indices

    hull_voxels = hull_voxel_indices(carve.grid, carve.centres)
    sigma = 0.0
    if np.any(hull_fog) and len(hull_voxels):
        totals = np.bincount(
            carve.grid.index_of(positions[hull_fog]),
            weights=cross_section[hull_fog],
            minlength=carve.grid.size,
        )
        sigma = float(totals[hull_voxels].sum() / len(hull_voxels) / carve.grid.voxel**3)
    lo, hi = carve.reference
    room_span = float(np.linalg.norm(hi - lo))
    return {
        "gaussians_inside_hull": int(np.count_nonzero(in_hull)),
        "fog_inside_hull": int(np.count_nonzero(hull_fog)),
        "fog_fraction_of_inside": float(
            np.count_nonzero(hull_fog) / max(np.count_nonzero(in_hull), 1)
        ),
        "extinction_per_unit": sigma,
        "mean_free_path": float(1.0 / sigma) if sigma > 0 else None,
        "transmittance_across_room": float(np.exp(-sigma * room_span)),
        "room_span": room_span,
    }


def compute_keep_mask(
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    *,
    outlier_std: float = 8.0,
    min_opacity: float = 0.01,
    max_scale: float | None = None,
    max_scale_frac: float | None = None,
    crop_margin: float | None = None,
    min_neighbors: int = 0,
    neighbor_radius_frac: float = 0.01,
    max_needle_ratio: float | None = None,
    needle_min_length_frac: float = 0.005,
    min_view_count: int = 0,
    view_counts: np.ndarray | None = None,
    observed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    free_space_remove: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build a boolean keep mask for Gaussians.

    Filters run cheapest-first: opacity, scale, observed-volume crop, local
    density, then the legacy global-radius rule. Scene scale is taken from
    ``observed_bounds`` when supplied (the SfM points), otherwise from a robust
    percentile span of the Gaussians themselves, so the fractional thresholds
    mean the same thing regardless of COLMAP's arbitrary units.
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3), got {pts.shape}")
    n = pts.shape[0]
    opacities = np.asarray(opacities, dtype=np.float64).reshape(n)
    scales = np.asarray(scales, dtype=np.float64)
    if scales.ndim == 1:
        scales = scales.reshape(n, -1)
    if scales.shape[0] != n:
        raise ValueError(f"scales length {scales.shape[0]} != positions {n}")

    keep = np.ones(n, dtype=bool)
    removed_opacity = 0
    removed_scale = 0
    removed_spatial = 0
    removed_crop = 0
    removed_density = 0
    removed_needle = 0
    removed_visibility = 0
    removed_free_space = 0

    if observed_bounds is not None:
        lo, hi = (np.asarray(observed_bounds[0], dtype=np.float64),
                  np.asarray(observed_bounds[1], dtype=np.float64))
    else:
        lo, hi = np.percentile(pts, [2.0, 98.0], axis=0)
    span = np.maximum(hi - lo, 1e-9)
    diagonal = float(np.linalg.norm(span))

    opacity_keep = opacities >= float(min_opacity)
    removed_opacity = int(np.count_nonzero(~opacity_keep))
    keep &= opacity_keep

    scale_limit = None
    if max_scale is not None:
        scale_limit = float(max_scale)
    if max_scale_frac is not None:
        frac_limit = float(max_scale_frac) * diagonal
        scale_limit = frac_limit if scale_limit is None else min(scale_limit, frac_limit)
    if scale_limit is not None:
        scale_keep = np.max(scales, axis=1) <= scale_limit
        removed_scale = int(np.count_nonzero(keep & ~scale_keep))
        keep &= scale_keep

    if max_needle_ratio is not None:
        sorted_axes = np.sort(scales, axis=1)
        needle = sorted_axes[:, -1] / np.maximum(sorted_axes[:, -2], 1e-9)
        long_enough = sorted_axes[:, -1] > float(needle_min_length_frac) * diagonal
        needle_keep = ~((needle > float(max_needle_ratio)) & long_enough)
        removed_needle = int(np.count_nonzero(keep & ~needle_keep))
        keep &= needle_keep

    if crop_margin is not None:
        pad = float(crop_margin) * span
        crop_keep = np.all((pts >= lo - pad) & (pts <= hi + pad), axis=1)
        removed_crop = int(np.count_nonzero(keep & ~crop_keep))
        keep &= crop_keep

    if min_neighbors > 0 and np.count_nonzero(keep) > int(min_neighbors):
        radius = float(neighbor_radius_frac) * diagonal
        idx = np.flatnonzero(keep)
        counts = _neighbor_counts(pts[idx], radius)
        dense_enough = counts >= int(min_neighbors)
        removed_density = int(np.count_nonzero(~dense_enough))
        keep[idx[~dense_enough]] = False

    if min_view_count > 0 and view_counts is not None:
        seen_enough = np.asarray(view_counts).reshape(n) >= int(min_view_count)
        removed_visibility = int(np.count_nonzero(keep & ~seen_enough))
        keep &= seen_enough

    if free_space_remove is not None:
        carve_keep = ~np.asarray(free_space_remove, dtype=bool).reshape(n)
        removed_free_space = int(np.count_nonzero(keep & ~carve_keep))
        keep &= carve_keep

    if float(outlier_std) > 0.0 and np.count_nonzero(keep) >= 8:
        candidates = pts[keep]
        center = np.median(candidates, axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        cand_dists = dists[keep]
        mad = float(np.median(np.abs(cand_dists - np.median(cand_dists))))
        sigma = 1.4826 * mad
        if sigma < 1e-9:
            sigma = float(np.std(cand_dists)) + 1e-9
        spatial_keep = dists <= (float(outlier_std) * sigma)
        removed_spatial = int(np.count_nonzero(keep & ~spatial_keep))
        keep &= spatial_keep

    return keep, {
        "removed_spatial": removed_spatial,
        "removed_opacity": removed_opacity,
        "removed_scale": removed_scale,
        "removed_crop": removed_crop,
        "removed_density": removed_density,
        "removed_needle": removed_needle,
        "removed_visibility": removed_visibility,
        "removed_free_space": removed_free_space,
        "scene_diagonal": diagonal,
    }


def filter_parallel_arrays(
    keep: np.ndarray,
    *,
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    orientations: np.ndarray,
    sh_coeffs: np.ndarray | None,
    sh_element_size: int,
) -> dict[str, np.ndarray]:
    """Apply a keep mask to all per-Gaussian arrays (including packed SH)."""
    keep = np.asarray(keep, dtype=bool)
    n = len(keep)
    out: dict[str, np.ndarray] = {
        "positions": np.asarray(positions)[keep],
        "opacities": np.asarray(opacities).reshape(n)[keep],
        "scales": np.asarray(scales)[keep],
        "orientations": np.asarray(orientations)[keep],
    }
    if sh_coeffs is not None:
        coeffs = np.asarray(sh_coeffs)
        if coeffs.ndim != 2 or coeffs.shape[1] != 3:
            raise ValueError(f"sh_coeffs must be (N*elementSize, 3), got {coeffs.shape}")
        if sh_element_size <= 0:
            raise ValueError("sh_element_size must be positive")
        expected = n * int(sh_element_size)
        if coeffs.shape[0] != expected:
            raise ValueError(
                f"sh_coeffs length {coeffs.shape[0]} != N*elementSize ({expected})"
            )
        packed = coeffs.reshape(n, int(sh_element_size), 3)
        out["sh_coeffs"] = packed[keep].reshape(-1, 3)
    return out


def _vt_to_numpy(values: Any) -> np.ndarray:
    """Convert USD Vt arrays / lists of Gf types into a dense numpy array."""
    if values is None:
        raise ValueError("attribute value is None")
    if hasattr(values, "__len__") and len(values) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    sample = values[0]
    # Quaternions: Gf.Quatf / Quath → (w, x, y, z)
    if hasattr(sample, "GetReal") and hasattr(sample, "GetImaginary"):
        rows = []
        for q in values:
            imag = q.GetImaginary()
            rows.append([float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])])
        return np.asarray(rows, dtype=np.float64)

    if hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
        return np.asarray([[float(x) for x in item] for item in values], dtype=np.float64)

    return np.asarray([float(x) for x in values], dtype=np.float64)


def _find_particle_field_prim(stage: Any) -> Any:
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        if "ParticleField" in type_name and prim.GetAttribute("positions").HasValue():
            return prim
    raise RuntimeError("No ParticleField with positions found in stage")


def _load_gaussian_arrays(prim: Any) -> dict[str, Any]:
    positions = _vt_to_numpy(prim.GetAttribute("positions").Get())
    opacities = _vt_to_numpy(prim.GetAttribute("opacities").Get())
    scales = _vt_to_numpy(prim.GetAttribute("scales").Get())
    orientations = _vt_to_numpy(prim.GetAttribute("orientations").Get())
    sh_attr = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
    sh_coeffs = None
    sh_element_size = 1
    if sh_attr and sh_attr.HasValue():
        sh_coeffs = _vt_to_numpy(sh_attr.Get())
        meta = sh_attr.GetMetadata("elementSize")
        sh_element_size = int(meta) if meta else max(1, sh_coeffs.shape[0] // positions.shape[0])
    return {
        "positions": positions,
        "opacities": opacities,
        "scales": scales,
        "orientations": orientations,
        "sh_coeffs": sh_coeffs,
        "sh_element_size": sh_element_size,
    }


def _write_gaussian_arrays(prim: Any, arrays: dict[str, np.ndarray], *, sh_element_size: int) -> None:
    from pxr import Gf, Vt, UsdGeom

    positions = np.asarray(arrays["positions"], dtype=np.float32)
    opacities = np.asarray(arrays["opacities"], dtype=np.float32).reshape(-1)
    scales = np.asarray(arrays["scales"], dtype=np.float32)
    orientations = np.asarray(arrays["orientations"], dtype=np.float64)

    prim.GetAttribute("positions").Set(Vt.Vec3fArray.FromNumpy(positions))
    prim.GetAttribute("opacities").Set(Vt.FloatArray.FromNumpy(opacities))
    prim.GetAttribute("scales").Set(Vt.Vec3fArray.FromNumpy(scales))

    quats = [
        Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in orientations
    ]
    prim.GetAttribute("orientations").Set(Vt.QuatfArray(quats))

    sh_attr = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
    if sh_attr and "sh_coeffs" in arrays:
        coeffs = np.asarray(arrays["sh_coeffs"], dtype=np.float32)
        sh_attr.Set(Vt.Vec3fArray.FromNumpy(coeffs))
        sh_attr.SetMetadata("elementSize", int(sh_element_size))

    if positions.shape[0] == 0:
        extent = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)]
    else:
        lo = positions.min(axis=0)
        hi = positions.max(axis=0)
        extent = [
            Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
            Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2])),
        ]
    extent_attr = prim.GetAttribute("extent")
    if extent_attr:
        extent_attr.Set(extent)
    else:
        UsdGeom.Boundable(prim).CreateExtentAttr().Set(extent)


def cleanup_particlefield_file(
    input_path: Path,
    output_path: Path,
    params: SplatCleanupParams,
    *,
    raw_backup_path: Path | None = None,
    observed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    view_counts: np.ndarray | None = None,
    colmap_txt_dir: Path | None = None,
    colmap_sparse_dir: Path | None = None,
) -> SplatCleanupReport:
    """Filter stray Gaussians in a ParticleField USD (requires OpenUSD ``pxr``)."""
    from pxr import Usd

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if raw_backup_path is not None:
        raw_backup_path = raw_backup_path.expanduser().resolve()
        raw_backup_path.parent.mkdir(parents=True, exist_ok=True)
        # Refresh raw backup from this input (e.g. cleaned→raw seed). Skip when the
        # input *is* already the raw backup (common GUI / re-cleanup path).
        if raw_backup_path != input_path:
            shutil.copy2(input_path, raw_backup_path)

    # Prefer reading from the fresh raw backup when doing an in-place filter so
    # we never open a half-written output path.
    if output_path.resolve() == input_path.resolve():
        if raw_backup_path is not None and raw_backup_path.is_file():
            work_path = raw_backup_path
        else:
            tmp = input_path.with_name(input_path.stem + "_cleanup_src" + input_path.suffix)
            shutil.copy2(input_path, tmp)
            work_path = tmp
    else:
        work_path = input_path

    if view_counts is None and colmap_txt_dir is not None:
        view_counts = view_counts_from_colmap(colmap_txt_dir, work_path)

    stage = Usd.Stage.Open(str(work_path))
    if stage is None:
        raise RuntimeError(f"Could not open ParticleField USD: {work_path}")
    prim = _find_particle_field_prim(stage)
    loaded = _load_gaussian_arrays(prim)

    free_space_remove = None
    carve_stats: dict[str, Any] | None = None
    fog_metrics: dict[str, Any] | None = None
    free_space_error: str | None = None
    carve = None
    if colmap_sparse_dir is not None:
        try:
            carve = _carve(loaded["positions"], Path(colmap_sparse_dir), params)
        except Exception as exc:  # noqa: BLE001
            # Never mis-carve on a frame mismatch or a missing model: the rule is
            # only meaningful if the Gaussians and the COLMAP points are in the
            # same space, and a wrong carve deletes real surfaces.
            free_space_error = f"{type(exc).__name__}: {exc}"
    elif params.free_space_votes > 0:
        free_space_error = (
            "free_space_votes is set but no COLMAP sparse dir was supplied; "
            "carving skipped"
        )

    if carve is not None and (params.free_space_votes > 0 or params.air_min_neighbors > 0):
        from scan2usd.reconstruction.free_space import free_space_removal_mask

        free_space_remove, carve_stats = free_space_removal_mask(
            loaded["positions"],
            carve,
            min_free_votes=params.free_space_votes,
            surface_radius_frac=params.surface_radius_frac,
            air_min_neighbors=params.air_min_neighbors,
            air_neighbor_radius_frac=params.air_neighbor_radius_frac,
            free_behind=params.free_behind,
            hull_air=params.hull_air,
        )

    keep, removed = compute_keep_mask(
        loaded["positions"],
        loaded["opacities"],
        loaded["scales"],
        outlier_std=params.outlier_std,
        min_opacity=params.min_opacity,
        max_scale=params.max_scale,
        max_scale_frac=params.max_scale_frac,
        crop_margin=params.crop_margin,
        min_neighbors=params.min_neighbors,
        neighbor_radius_frac=params.neighbor_radius_frac,
        max_needle_ratio=params.max_needle_ratio,
        needle_min_length_frac=params.needle_min_length_frac,
        min_view_count=params.min_view_count,
        view_counts=view_counts,
        free_space_remove=free_space_remove,
        observed_bounds=observed_bounds,
    )
    filtered = filter_parallel_arrays(
        keep,
        positions=loaded["positions"],
        opacities=loaded["opacities"],
        scales=loaded["scales"],
        orientations=loaded["orientations"],
        sh_coeffs=loaded["sh_coeffs"],
        sh_element_size=int(loaded["sh_element_size"]),
    )
    _write_gaussian_arrays(
        prim,
        filtered,
        sh_element_size=int(loaded["sh_element_size"]),
    )
    kept = int(np.count_nonzero(keep))
    total = int(loaded["positions"].shape[0])
    if params.min_keep_fraction > 0 and total and kept / total < params.min_keep_fraction:
        raise RuntimeError(
            f"Cleanup would keep only {kept:,}/{total:,} Gaussians "
            f"({kept / total:.2%}, floor {params.min_keep_fraction:.0%}). "
            f"Removed by: opacity={removed['removed_opacity']:,} "
            f"scale={removed['removed_scale']:,} crop={removed['removed_crop']:,} "
            f"needle={removed['removed_needle']:,} density={removed['removed_density']:,}. "
            "If opacity dominates, the trained model is probably degenerate rather "
            "than dirty — check the training schedule before lowering min_opacity."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output_path)):
        raise RuntimeError(f"Failed to export cleaned ParticleField USD: {output_path}")

    if work_path != input_path and work_path.name.endswith("_cleanup_src" + input_path.suffix):
        work_path.unlink(missing_ok=True)

    # Measure the result, not the intent: fog after filtering is what the scene
    # actually ships with, and it is the one number held-out PSNR cannot supply.
    if carve is not None:
        try:
            fog_metrics = fog_metrics_from_carve(
                carve,
                loaded["positions"][keep],
                np.asarray(loaded["opacities"]).reshape(-1)[keep],
                np.asarray(loaded["scales"])[keep],
                params,
            )
        except Exception as exc:  # noqa: BLE001
            free_space_error = f"fog metrics unavailable: {type(exc).__name__}: {exc}"

    return SplatCleanupReport(
        input_count=int(loaded["positions"].shape[0]),
        kept_count=int(np.count_nonzero(keep)),
        removed_spatial=removed["removed_spatial"],
        removed_opacity=removed["removed_opacity"],
        removed_scale=removed["removed_scale"],
        outlier_std=float(params.outlier_std),
        min_opacity=float(params.min_opacity),
        max_scale=None if params.max_scale is None else float(params.max_scale),
        input_path=str(input_path),
        output_path=str(output_path),
        raw_backup_path=None if raw_backup_path is None else str(raw_backup_path),
        removed_crop=removed["removed_crop"],
        removed_density=removed["removed_density"],
        removed_needle=removed["removed_needle"],
        removed_visibility=removed["removed_visibility"],
        removed_free_space=removed["removed_free_space"],
        free_space_breakdown=carve_stats,
        fog_metrics=fog_metrics,
        free_space_error=free_space_error,
        scene_diagonal=removed["scene_diagonal"],
    )


def view_counts_from_colmap(colmap_txt_dir: Path, splat_path: Path) -> np.ndarray | None:
    """View counts for the Gaussians in ``splat_path`` using the COLMAP cameras."""
    from pxr import Usd

    from scan2usd.reconstruction.colmap_io import parse_cameras_txt, parse_images_txt

    colmap_txt_dir = Path(colmap_txt_dir)
    cameras_txt = colmap_txt_dir / "cameras.txt"
    images_txt = colmap_txt_dir / "images.txt"
    if not (cameras_txt.is_file() and images_txt.is_file()):
        return None
    cameras = parse_cameras_txt(cameras_txt)
    images = parse_images_txt(images_txt)
    if not cameras or not images:
        return None
    stage = Usd.Stage.Open(str(splat_path))
    prim = _find_particle_field_prim(stage)
    positions = _vt_to_numpy(prim.GetAttribute("positions").Get())
    return compute_view_counts(positions, cameras, images)


def observed_bounds_from_colmap(
    cfg: Any,
    *,
    percentile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Robust bounds of the reconstructed scene from the COLMAP sparse model.

    Percentile-trimmed *and* unioned with the camera positions, because trimming
    alone is not enough: on the bedroom the 2-98 percentile of SfM points still
    spans 22 units against a 10-unit room, and the 1-99 percentile spans 69.7.
    Camera centres have no such tail — the operator physically stood there — and
    their bounds match the 5-95 point extent to within a few percent. Using the
    looser box let `crop_margin` keep a halo the whole width of the scene.
    """
    from scan2usd.reconstruction.colmap_io import parse_images_txt, parse_points3d_txt

    txt_dir = Path(getattr(cfg, "colmap_txt_dir", ""))
    points_txt = txt_dir / "points3D.txt"
    if not points_txt.is_file():
        return None
    _ids, points = parse_points3d_txt(points_txt)
    if len(points) < 32:
        return None
    lo, hi = np.percentile(points, [percentile, 100.0 - percentile], axis=0)

    images_txt = txt_dir / "images.txt"
    if images_txt.is_file():
        centres = [pose.camera_center() for pose in parse_images_txt(images_txt).values()]
        if centres:
            centre_arr = np.asarray(centres, dtype=np.float64)
            lo = np.minimum(lo, centre_arr.min(axis=0))
            hi = np.maximum(hi, centre_arr.max(axis=0))
    return np.asarray(lo), np.asarray(hi)


def cleanup_particlefield_via_isaac(
    cfg: Any,
    input_path: Path,
    output_path: Path,
    params: SplatCleanupParams,
    *,
    raw_backup_path: Path | None = None,
    report_path: Path | None = None,
    observed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    colmap_dir: Path | None = None,
    colmap_sparse_dir: Path | None = None,
) -> SplatCleanupReport:
    """Run cleanup under Isaac's Python (has OpenUSD ParticleField schemas)."""
    from scan2usd.reconstruction.external_cli import ExternalToolAdapter, resolve_external_command

    prefix = resolve_external_command(
        cfg,
        "isaac_python",
        default="python.sh",
        required=True,
    )
    script = Path(__file__).resolve().parents[3] / "tools" / "geometry" / "cleanup_splat_usd.py"
    report_path = report_path or (output_path.parent / "splat_cleanup_report.json")
    args = [
        str(script),
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_path.resolve()),
        "--report",
        str(report_path.resolve()),
        "--outlier-std",
        str(params.outlier_std),
        "--min-opacity",
        str(params.min_opacity),
        "--min-neighbors",
        str(params.min_neighbors),
        "--neighbor-radius-frac",
        str(params.neighbor_radius_frac),
        "--needle-min-length-frac",
        str(params.needle_min_length_frac),
        "--min-view-count",
        str(params.min_view_count),
    ]
    if colmap_dir is not None:
        args.extend(["--colmap-txt", str(Path(colmap_dir).resolve())])
    if colmap_sparse_dir is not None:
        args.extend(["--colmap-sparse", str(Path(colmap_sparse_dir).resolve())])
    args.extend(
        [
            "--free-space-votes",
            str(params.free_space_votes),
            "--carve-resolution",
            str(params.carve_resolution),
            "--carve-max-rays",
            str(params.carve_max_rays),
            "--surface-radius-frac",
            str(params.surface_radius_frac),
            "--air-min-neighbors",
            str(params.air_min_neighbors),
            "--air-neighbor-radius-frac",
            str(params.air_neighbor_radius_frac),
        ]
    )
    if params.free_behind:
        args.append("--free-behind")
    if params.hull_air:
        args.append("--hull-air")
    if params.max_needle_ratio is not None:
        args.extend(["--max-needle-ratio", str(params.max_needle_ratio)])
    if params.max_scale is not None:
        args.extend(["--max-scale", str(params.max_scale)])
    if params.max_scale_frac is not None:
        args.extend(["--max-scale-frac", str(params.max_scale_frac)])
    if params.crop_margin is not None:
        args.extend(["--crop-margin", str(params.crop_margin)])
    if observed_bounds is not None:
        args.extend(["--observed-min", *[str(v) for v in observed_bounds[0]]])
        args.extend(["--observed-max", *[str(v) for v in observed_bounds[1]]])
    if raw_backup_path is not None:
        args.extend(["--raw-backup", str(raw_backup_path.resolve())])
    adapter = ExternalToolAdapter("isaac_python", prefix)
    adapter.run(*args)
    if not report_path.is_file():
        raise RuntimeError(f"Cleanup did not write report: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return SplatCleanupReport(**payload)


def cleanup_particlefield(
    cfg: Any,
    input_path: Path,
    output_path: Path,
    params: SplatCleanupParams | None = None,
    *,
    raw_backup_path: Path | None = None,
) -> SplatCleanupReport:
    """
    Clean a ParticleField USD using in-process ``pxr`` when available, else Isaac.
    """
    params = params or SplatCleanupParams()
    observed_bounds = None
    if params.crop_margin is not None or params.max_scale_frac is not None:
        observed_bounds = observed_bounds_from_colmap(cfg)
    colmap_dir = Path(getattr(cfg, "colmap_txt_dir", "")) if params.min_view_count > 0 else None
    # Carving and the fog metrics need per-point tracks (which cameras observed
    # each point); only the binary sparse model carries those, not the TXT export.
    sparse_dir = Path(getattr(cfg, "nerfstudio_data_dir", "")) / "colmap" / "sparse" / "0"
    colmap_sparse_dir = sparse_dir if (sparse_dir / "points3D.bin").is_file() else None
    if not params.enabled:
        if output_path.resolve() != input_path.resolve():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
        return SplatCleanupReport(
            input_count=-1,
            kept_count=-1,
            removed_spatial=0,
            removed_opacity=0,
            removed_scale=0,
            outlier_std=params.outlier_std,
            min_opacity=params.min_opacity,
            max_scale=params.max_scale,
            input_path=str(input_path),
            output_path=str(output_path),
            raw_backup_path=None,
        )
    try:
        import pxr  # noqa: F401

        return cleanup_particlefield_file(
            input_path,
            output_path,
            params,
            raw_backup_path=raw_backup_path,
            observed_bounds=observed_bounds,
            colmap_txt_dir=colmap_dir,
            colmap_sparse_dir=colmap_sparse_dir,
        )
    except ImportError:
        return cleanup_particlefield_via_isaac(
            cfg,
            input_path,
            output_path,
            params,
            raw_backup_path=raw_backup_path,
            observed_bounds=observed_bounds,
            colmap_dir=colmap_dir,
            colmap_sparse_dir=colmap_sparse_dir,
        )


def write_report_json(report: SplatCleanupReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
