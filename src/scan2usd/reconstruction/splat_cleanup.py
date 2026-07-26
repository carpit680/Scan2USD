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
    """Thresholds for stray-Gaussian cleanup."""

    enabled: bool = True
    outlier_std: float = 4.0
    min_opacity: float = 0.01
    max_scale: float | None = None


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_keep_mask(
    positions: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    *,
    outlier_std: float = 4.0,
    min_opacity: float = 0.01,
    max_scale: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build a boolean keep mask for Gaussians.

    Spatial rule: keep points within ``outlier_std * sigma_mad`` of the median
    center, where ``sigma_mad = 1.4826 * MAD`` (robust σ estimate).
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

    opacity_keep = opacities >= float(min_opacity)
    removed_opacity = int(np.count_nonzero(~opacity_keep))
    keep &= opacity_keep

    if max_scale is not None:
        scale_keep = np.max(scales, axis=1) <= float(max_scale)
        removed_scale = int(np.count_nonzero(keep & ~scale_keep))
        keep &= scale_keep

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
    )


def cleanup_particlefield_via_isaac(
    cfg: Any,
    input_path: Path,
    output_path: Path,
    params: SplatCleanupParams,
    *,
    raw_backup_path: Path | None = None,
    report_path: Path | None = None,
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
    ]
    if params.max_scale is not None:
        args.extend(["--max-scale", str(params.max_scale)])
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
        )
    except ImportError:
        return cleanup_particlefield_via_isaac(
            cfg,
            input_path,
            output_path,
            params,
            raw_backup_path=raw_backup_path,
        )


def write_report_json(report: SplatCleanupReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
