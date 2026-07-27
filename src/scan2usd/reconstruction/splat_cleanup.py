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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(output_path)):
        raise RuntimeError(f"Failed to export cleaned ParticleField USD: {output_path}")

    if work_path != input_path and work_path.name.endswith("_cleanup_src" + input_path.suffix):
        work_path.unlink(missing_ok=True)

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
    percentile: float = 2.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Robust bounds of the reconstructed scene from the COLMAP sparse points.

    Percentile-trimmed because SfM always leaves a few wild points, and using the
    raw min/max would define an "observed volume" many times the real room.
    """
    points_txt = Path(getattr(cfg, "colmap_txt_dir", "")) / "points3D.txt"
    if not points_txt.is_file():
        return None
    from scan2usd.reconstruction.colmap_io import parse_points3d_txt

    _ids, points = parse_points3d_txt(points_txt)
    if len(points) < 32:
        return None
    lo, hi = np.percentile(points, [percentile, 100.0 - percentile], axis=0)
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
        )


def write_report_json(report: SplatCleanupReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
