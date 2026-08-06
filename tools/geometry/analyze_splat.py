#!/usr/bin/env python
"""
Diagnose *where* a Gaussian model puts its mass, and how much of it is fog.

Held-out PSNR answers "do the training views reproduce?" It cannot answer "is the
air inside the room clear?", because haze that sits between the camera and a wall
is often photometrically almost free — it reproduces the same pixels the wall
would. That is exactly the artifact a person notices the moment they stand inside
the scene, and it is what this tool measures.

Three questions, in order of how much they assume:

1. **Population** — opacity, scale, anisotropy distributions. No assumptions.
2. **Free space** — which Gaussians sit in volume the cameras looked *through*.
   Every (camera, SfM point) pair is a ray whose interior is known-empty: the
   camera saw that point, so nothing opaque was in the way. Voxels along those
   segments are carved free; voxels holding SfM points are marked occupied.
   A Gaussian in a carved-free voxel with no occupancy is fog by construction,
   not by threshold.
3. **Extinction** — how opaque the free space actually is. A Gaussian of opacity
   ``a`` and radius ``s`` blocks a cross-section ``a*pi*s^2``; summed over a voxel
   of side ``h`` and divided by ``h^3`` that is an extinction coefficient in
   units of 1/length, so ``T(d) = exp(-sigma*d)`` is the fraction of a view that
   survives ``d`` units of travel. That number is the user-visible "cloudiness",
   and it is comparable across scenes and across cleanup settings.

Run under Isaac's Python (it has ``pxr``)::

    workspace/isaac_env/bin/python tools/geometry/analyze_splat.py \
        --stage workspace_bedroom/build/visual/environment_splat.usd \
        --colmap workspace_bedroom/ns_data/colmap/sparse/0 \
        --out workspace_bedroom/build/visual/splat_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scan2usd.reconstruction.free_space import (  # noqa: E402
    OBSERVED_DILATION,
    SURFACE_RADIUS_FRAC,
    build_grid,
    carve_free_space,
    hull_voxel_indices,
    inside_camera_hull,
    observed_reference_bounds,
    read_images_bin,
    read_points3d_bin,
    require_same_frame,
    within_surface_radius,
)

# ------------------------------------------------------------------ USD splat


def load_gaussians(stage_path: Path) -> dict[str, np.ndarray]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")
    prim = None
    for candidate in stage.Traverse():
        type_name = candidate.GetTypeName()
        if "ParticleField" in str(type_name) and candidate.GetAttribute("positions").HasValue():
            prim = candidate
            break
    if prim is None:
        raise RuntimeError(
            f"No ParticleField3DGaussianSplat prim with positions in {stage_path}. "
            "A NuRec export hides Gaussians inside an opaque field asset and cannot "
            "be analysed; re-export with reconstruction.usd_splat_format: standard."
        )
    positions = np.asarray(prim.GetAttribute("positions").Get(), dtype=np.float64)
    opacities = np.asarray(prim.GetAttribute("opacities").Get(), dtype=np.float64).reshape(-1)
    scales = np.asarray(prim.GetAttribute("scales").Get(), dtype=np.float64)
    return {"positions": positions, "opacities": opacities, "scales": scales}


# ------------------------------------------------------------------- report


def percentiles(values: np.ndarray, points=(1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(values, p)) for p in points}


def analyze(
    stage: Path,
    colmap_dir: Path,
    *,
    resolution: int,
    max_rays: int,
    min_free_votes: int,
    surface_radius_frac: float = SURFACE_RADIUS_FRAC,
) -> dict:
    gaussians = load_gaussians(stage)
    positions = gaussians["positions"]
    opacities = gaussians["opacities"]
    scales = gaussians["scales"]

    # Authored Gaussian positions are in COLMAP space: 3DGRUT exports them there
    # and the pipeline only ever adds an xformOp on the root stage, so the
    # analysis stays in COLMAP space too and nothing needs transforming. The
    # check below is what makes that a verified fact rather than an assumption —
    # comparing a USD-frame point cloud against COLMAP-frame Gaussians produces
    # plausible-looking numbers that are entirely meaningless.
    centres = read_images_bin(colmap_dir / "images.bin")
    points, tracks = read_points3d_bin(colmap_dir / "points3D.bin")
    require_same_frame(positions, points)

    extents = positions.max(axis=0) - positions.min(axis=0)
    diagonal = float(np.linalg.norm(extents))
    scale_mid = np.sort(scales, axis=1)[:, 1] if scales.ndim == 2 else scales
    scale_max = scales.max(axis=1) if scales.ndim == 2 else scales
    scale_min = scales.min(axis=1) if scales.ndim == 2 else scales

    # Grid anchored to the observed volume, so voxel size is a property of the
    # capture rather than of whichever model is being scored.
    centre_array = np.asarray(list(centres.values()), dtype=np.float64)
    reference = observed_reference_bounds(points, centre_array)
    grid = build_grid(reference, resolution, dilation=OBSERVED_DILATION)
    free, occupied = carve_free_space(
        grid,
        centres,
        points,
        tracks,
        max_rays=max_rays,
        stop_margin=3.0 * grid.voxel,
    )

    voxel_of = grid.index_of(positions)
    in_grid = grid.contains(positions)
    # Beyond the observed volume entirely — the exterior halo. Counted, but kept
    # out of every interior statistic; nobody stands out there.
    is_outside = ~in_grid

    # "On a surface" is a metric question, not a voxel one. Defining it as
    # "shares a voxel with an SfM point" makes the answer track the grid: at
    # 0.35-unit voxels 82% of the bedroom reads as surface, at 0.06-unit voxels
    # only 71% does, because SfM points are sparser than a fine grid. A fixed
    # radius is stable under any resolution.
    surface_radius = surface_radius_frac * float(np.linalg.norm(reference[1] - reference[0]))
    near_surface = within_surface_radius(positions, points, surface_radius)

    is_surface = in_grid & near_surface
    carved_free = in_grid & (free[voxel_of] >= min_free_votes)
    # Fog needs both: the cameras saw through here, AND no surface is near. A
    # featureless white wall has few SfM points, but nothing is behind it to aim
    # rays at either, so it fails the first test and is never called fog.
    is_free = carved_free & ~near_surface
    is_unobserved = in_grid & ~near_surface & ~carved_free

    # Blocking cross-section: what a Gaussian actually hides from a viewer.
    cross_section = opacities * np.pi * scale_mid**2
    voxel_volume = grid.voxel**3

    free_voxels = np.nonzero((free >= min_free_votes) & (occupied == 0))[0]

    def extinction(mask: np.ndarray, over_voxels: np.ndarray) -> float:
        """
        Mean extinction (1/length) a ray meets crossing ``over_voxels``.

        Averaged over every voxel in the region, not only the ones holding
        Gaussians: a ray traverses the empty ones too, and conditioning on
        occupied voxels would overstate the haze by the region's fill factor.
        """
        if not np.any(mask) or not len(over_voxels):
            return 0.0
        totals = np.bincount(voxel_of[mask], weights=cross_section[mask], minlength=grid.size)
        return float(totals[over_voxels].sum() / len(over_voxels) / voxel_volume)

    occupied_voxels = np.nonzero(occupied > 0)[0]
    fog_sigma = extinction(is_free, free_voxels)

    # The volume the operator physically walked through. Anything in here that is
    # not a surface is what makes the room look cloudy from the inside; the halo
    # outside the walls is a separate problem and must not be averaged in.
    centre_array = np.asarray(list(centres.values()), dtype=np.float64)
    in_hull = inside_camera_hull(positions, centre_array) & in_grid
    hull_voxels = hull_voxel_indices(grid, centre_array)
    # Haze is everything in the walked volume with no surface near it, whether
    # or not a ray happened to prove its cell empty. Measuring only the carved
    # subset would report a clear room the moment the carve filter had run.
    hull_fog = in_hull & ~near_surface
    hull_carved = in_hull & is_free
    hull_sigma = extinction(hull_fog, hull_voxels)

    # Robust span: raw COLMAP min/max includes a few wildly misplaced points, so
    # the 5-95 percentile extent is what an occupant actually looks across.
    lo, hi = np.percentile(points, [5, 95], axis=0)
    room_span = float(np.linalg.norm(hi - lo))
    camera_span = float(
        np.linalg.norm(centre_array.max(axis=0) - centre_array.min(axis=0))
    )

    report = {
        "stage": str(stage),
        "gaussians": int(len(positions)),
        "bounds_min": positions.min(axis=0).tolist(),
        "bounds_max": positions.max(axis=0).tolist(),
        "extents": extents.tolist(),
        "diagonal": diagonal,
        # Grid geometry is fixed by the capture, not the model, so these are
        # identical for every model of this scene and the numbers are comparable.
        "voxel": grid.voxel,
        "grid_dims": grid.dims.tolist(),
        "grid_origin": grid.origin.tolist(),
        "observed_dilation": OBSERVED_DILATION,
        "cameras": len(centres),
        "sfm_points": int(len(points)),
        "opacity": percentiles(opacities),
        "scale_mid": percentiles(scale_mid),
        "anisotropy": percentiles(scale_max / np.maximum(scale_min, 1e-9)),
        "population": {
            "surface": int(np.count_nonzero(is_surface)),
            "free_space": int(np.count_nonzero(is_free)),
            "unobserved": int(np.count_nonzero(is_unobserved)),
            "outside_observed": int(np.count_nonzero(is_outside)),
            "free_space_fraction": float(np.count_nonzero(is_free) / len(positions)),
            "outside_observed_fraction": float(np.count_nonzero(is_outside) / len(positions)),
        },
        "blocking_cross_section": {
            "total": float(cross_section.sum()),
            "surface": float(cross_section[is_surface].sum()),
            "free_space": float(cross_section[is_free].sum()),
            "unobserved": float(cross_section[is_unobserved].sum()),
            "outside_observed": float(cross_section[is_outside].sum()),
            "free_space_fraction": float(
                cross_section[is_free].sum() / max(cross_section.sum(), 1e-12)
            ),
        },
        "camera_hull": {
            "gaussians_inside": int(np.count_nonzero(in_hull)),
            "fog_inside": int(np.count_nonzero(hull_fog)),
            "fog_fraction_of_inside": float(
                np.count_nonzero(hull_fog) / max(np.count_nonzero(in_hull), 1)
            ),
            "fog_cross_section": float(cross_section[hull_fog].sum()),
            "fog_cross_section_fraction": float(
                cross_section[hull_fog].sum() / max(cross_section.sum(), 1e-12)
            ),
            # Upper bound: everything in there with no surface nearby, whether or
            # not a ray happened to carve its voxel. fog_inside is the actionable
            # subset — what a free-space filter could defensibly delete.
            # The subset a free-space filter could defensibly delete today;
            # the rest is haze the ray coverage cannot yet prove empty.
            "carved_fog_inside": int(np.count_nonzero(hull_carved)),
            "camera_span": camera_span,
        },
        "extinction_per_unit": {
            "carved_free_space": fog_sigma,
            "camera_hull_fog": hull_sigma,
            "surface": extinction(is_surface, occupied_voxels),
        },
        "visibility": {
            # What a viewer standing in the room actually loses to haze.
            "free_voxels": int(len(free_voxels)),
            "occupied_voxels": int(len(occupied_voxels)),
            "hull_voxels": int(len(hull_voxels)),
            "mean_free_path": float(1.0 / hull_sigma) if hull_sigma > 0 else float("inf"),
            "transmittance_across_camera_path": float(np.exp(-hull_sigma * camera_span)),
            "transmittance_across_room": float(np.exp(-hull_sigma * room_span)),
            "room_span": room_span,
        },
        # What a free-space filter would actually do, at several strictnesses.
        # "kept_surface_loss" is the cost side: Gaussians that sit in a voxel
        # holding an SfM point, i.e. demonstrably on a surface, that the rule
        # would delete anyway. A usable rule keeps that at zero.
        "removal_preview": [
            {
                "min_free_votes": int(votes),
                "removed": int(np.count_nonzero(rule)),
                "removed_fraction": float(np.count_nonzero(rule) / len(positions)),
                "removed_inside_hull": int(np.count_nonzero(rule & in_hull)),
                "surface_loss": int(np.count_nonzero(rule & near_surface)),
                "cross_section_removed_fraction": float(
                    cross_section[rule].sum() / max(cross_section.sum(), 1e-12)
                ),
            }
            for votes in (1, 3, 10, 30, 100)
            for rule in [in_grid & (free[voxel_of] >= votes) & ~near_surface]
        ],
        "surface_population_stats": {
            "opacity": percentiles(opacities[is_surface]) if np.any(is_surface) else {},
            "scale_mid": percentiles(scale_mid[is_surface]) if np.any(is_surface) else {},
            "anisotropy": (
                percentiles((scale_max / np.maximum(scale_min, 1e-9))[is_surface])
                if np.any(is_surface)
                else {}
            ),
        },
        "fog_population_stats": {
            "opacity": percentiles(opacities[hull_fog]) if np.any(hull_fog) else {},
            "scale_mid": percentiles(scale_mid[hull_fog]) if np.any(hull_fog) else {},
            "anisotropy": (
                percentiles((scale_max / np.maximum(scale_min, 1e-9))[hull_fog])
                if np.any(hull_fog)
                else {}
            ),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--colmap", required=True, type=Path, help="COLMAP sparse/0 with .bin")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-rays", type=int, default=400_000)
    parser.add_argument("--min-free-votes", type=int, default=3)
    parser.add_argument("--surface-radius-frac", type=float, default=SURFACE_RADIUS_FRAC)
    args = parser.parse_args()

    report = analyze(
        args.stage,
        args.colmap,
        resolution=args.resolution,
        max_rays=args.max_rays,
        min_free_votes=args.min_free_votes,
        surface_radius_frac=args.surface_radius_frac,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)

    pop = report["population"]
    mass = report["blocking_cross_section"]
    hull = report["camera_hull"]
    vis = report["visibility"]
    print(
        "\n"
        f"  Surface / free / unobserved : {pop['surface']:,} / {pop['free_space']:,} / "
        f"{pop['unobserved']:,}\n"
        f"  Outside observed volume     : {pop['outside_observed']:,} "
        f"({pop['outside_observed_fraction'] * 100:.1f}%), "
        f"{mass['outside_observed'] / max(mass['total'], 1e-9) * 100:.1f}% of blocking mass\n"
        f"  Fog inside the camera path  : {hull['fog_inside']:,} Gaussians "
        f"({hull['fog_fraction_of_inside'] * 100:.1f}% of what is in there)\n"
        f"  Seen across the camera path : "
        f"{vis['transmittance_across_camera_path'] * 100:.1f}% transmittance\n"
        f"  Seen across the room        : "
        f"{vis['transmittance_across_room'] * 100:.1f}% transmittance",
        flush=True,
    )


if __name__ == "__main__":
    main()
