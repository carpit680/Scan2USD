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
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The analysis grid spans the observed volume padded by this fraction of its
# extent — enough to cover wall thickness and the Gaussians just behind surfaces.
OBSERVED_DILATION = 0.25
# A Gaussian within this fraction of the observed diagonal of an SfM point is
# carrying a surface. Resolution-independent, unlike voxel occupancy.
SURFACE_RADIUS_FRAC = 0.015

# ---------------------------------------------------------------- COLMAP (bin)


def _read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


def read_images_bin(path: Path) -> dict[int, np.ndarray]:
    """image_id -> camera centre in COLMAP world coordinates."""
    centres: dict[int, np.ndarray] = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = int(props[0])
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            name = b""
            while True:
                char = fid.read(1)
                if char == b"\x00" or char == b"":
                    break
                name += char
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2d)  # x, y, point3D_id per 2D point
            w, x, y, z = qvec
            rot = np.array(
                [
                    [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
                ]
            )
            centres[image_id] = -rot.T @ tvec
    return centres


def read_points3d_bin(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """(xyz array, per-point array of observing image ids)."""
    xyz: list[np.ndarray] = []
    tracks: list[np.ndarray] = []
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, 43, "QdddBBBd")
            xyz.append(np.array(props[1:4], dtype=np.float64))
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            raw = np.frombuffer(fid.read(8 * track_length), dtype=np.int32)
            tracks.append(raw[0::2].astype(np.int64))
    return np.asarray(xyz, dtype=np.float64), tracks


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


# -------------------------------------------------------------- free space


@dataclass
class Grid:
    origin: np.ndarray
    voxel: float
    dims: np.ndarray

    def index_of(self, points: np.ndarray) -> np.ndarray:
        """Flat voxel index, clamped. Only meaningful where ``contains`` is true."""
        idx = np.floor((points - self.origin) / self.voxel).astype(np.int64)
        np.clip(idx, 0, self.dims - 1, out=idx)
        return idx[:, 0] * (self.dims[1] * self.dims[2]) + idx[:, 1] * self.dims[2] + idx[:, 2]

    def contains(self, points: np.ndarray) -> np.ndarray:
        idx = np.floor((points - self.origin) / self.voxel).astype(np.int64)
        return np.all((idx >= 0) & (idx < self.dims), axis=1)

    @property
    def size(self) -> int:
        return int(np.prod(self.dims))


def observed_reference_bounds(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """
    The volume actually captured, robust to COLMAP's outlier points.

    Camera positions are the trustworthy part: they are where the operator
    physically stood, and unlike triangulated points they have no long tail. On
    the bedroom the camera bounds (7.9 x 10.4 x 5.5) match the 5-95 percentile
    SfM extent (7.7 x 10.1 x 5.0) to within a few percent, while the 1-99
    percentile still spans 69.7 on one axis and the raw min/max spans 170.
    Sizing a grid off those tails makes every voxel 6x too coarse.

    Union of the two so surfaces the cameras looked at are covered as well as
    the volume they moved through.
    """
    lo_pts, hi_pts = np.percentile(points, [5, 95], axis=0)
    lo = np.minimum(lo_pts, centres.min(axis=0))
    hi = np.maximum(hi_pts, centres.max(axis=0))
    return np.vstack([lo, hi])


def build_grid(points: np.ndarray, resolution: int, *, dilation: float = 0.0) -> Grid:
    """
    Grid spanning ``points``, optionally padded by ``dilation`` of its extent.

    Callers must pass a *stable reference* — the SfM points — not the Gaussians.
    Sizing the grid to the Gaussians makes every number here incomparable
    between two models of the same room: the bedroom's raw export spans 574
    units because of a handful of stray Gaussians, giving a 1.36-unit voxel, so
    the entire room interior collapses into 57 voxels and almost everything
    lands in a voxel that happens to hold an SfM point. The cleaned model of the
    same room, spanning 47 units, gets a 0.15-unit voxel and 48,832. Comparing
    the two measures the grid, not the fog.
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    if dilation:
        pad = (hi - lo) * float(dilation)
        lo = lo - pad
        hi = hi + pad
    voxel = float(np.max(hi - lo)) / float(resolution)
    # One extra layer: a point exactly on the upper bound floors to index
    # ceil(extent/voxel), which would otherwise be one past the last voxel and
    # count as outside the volume it defines.
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64), 1) + 1
    return Grid(origin=lo, voxel=voxel, dims=dims)


def carve_free_space(
    grid: Grid,
    centres: dict[int, np.ndarray],
    points: np.ndarray,
    tracks: list[np.ndarray],
    *,
    max_rays: int,
    stop_margin: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (free_votes, occupied_votes) per voxel, both uint16-saturated."""
    rng = np.random.default_rng(seed)

    origins: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for point_index, track in enumerate(tracks):
        for image_id in track:
            centre = centres.get(int(image_id))
            if centre is not None:
                origins.append(centre)
                targets.append(points[point_index])
    origins_arr = np.asarray(origins, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)

    # Rays aimed at COLMAP's outlier points leave the observed volume entirely;
    # their samples would all be discarded anyway, and their length sets the
    # step count for every chunk. Drop them before budgeting.
    on_target = grid.contains(targets_arr)
    origins_arr, targets_arr = origins_arr[on_target], targets_arr[on_target]

    total = len(origins_arr)
    if total > max_rays:
        pick = rng.choice(total, size=max_rays, replace=False)
        origins_arr = origins_arr[pick]
        targets_arr = targets_arr[pick]

    free = np.zeros(grid.size, dtype=np.uint32)
    occupied = np.zeros(grid.size, dtype=np.uint32)

    in_grid = grid.contains(points)
    np.add.at(occupied, grid.index_of(points[in_grid]), 1)

    direction = targets_arr - origins_arr
    lengths = np.linalg.norm(direction, axis=1)
    keep = lengths > (2.0 * grid.voxel + stop_margin)
    origins_arr, targets_arr = origins_arr[keep], targets_arr[keep]
    direction, lengths = direction[keep], lengths[keep]

    # One sample per voxel along the longest ray; shorter rays mask out the tail.
    steps = int(np.ceil(float(lengths.max()) / grid.voxel)) if len(lengths) else 0
    steps = max(1, min(steps, 4 * int(np.max(grid.dims))))
    chunk = max(1, int(4_000_000 / max(steps, 1)))

    for start in range(0, len(origins_arr), chunk):
        o = origins_arr[start : start + chunk]
        d = direction[start : start + chunk]
        length = lengths[start : start + chunk][:, None]
        # Stop short of the surface so the point's own Gaussians are not carved.
        limit = np.maximum(length - stop_margin, 0.0)
        t = (np.arange(steps, dtype=np.float64)[None, :] + 0.5) * grid.voxel
        valid = t < limit
        frac = np.where(valid, t / length, 0.0)
        samples = o[:, None, :] + frac[:, :, None] * d[:, None, :]
        flat = samples.reshape(-1, 3)[valid.reshape(-1)]
        flat = flat[grid.contains(flat)]  # never clamp a sample into an edge voxel
        if len(flat):
            # True ray-hit counts, so min_free_votes means "N rays passed through
            # here" rather than the meaningless "hit in N different chunks".
            free += np.bincount(grid.index_of(flat), minlength=grid.size).astype(np.uint32)
    return free, occupied


def inside_camera_hull(positions: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """
    Mask of Gaussians inside the convex hull of the camera path.

    The strongest available statement about interior fog: the operator physically
    walked the phone through this volume, so it is air. Anything dense in here is
    an artifact regardless of how it scores photometrically, and unlike the
    carving it needs no ray budget to be confident.
    """
    hull = _hull_of(centres)
    if hull is None:
        return np.zeros(len(positions), dtype=bool)
    return _inside(hull, positions)


def _hull_of(centres: np.ndarray):
    from scipy.spatial import Delaunay
    from scipy.spatial import QhullError

    if len(centres) < 4:
        return None
    # A capture panned at a fixed height is coplanar and has no volume, which
    # Qhull refuses outright. Joggling only produces a zero-thickness sliver that
    # then contains nothing, so give the sweep an explicit thickness instead:
    # the operator did occupy a slab, not a mathematical plane.
    centred = centres - centres.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    if len(singular) >= 3 and singular[2] < 1e-3 * max(singular[0], 1e-12):
        normal = np.linalg.svd(centred, full_matrices=True)[2][2]
        thickness = 0.02 * float(singular[0]) or 1e-6
        centres = np.vstack([centres + normal * thickness, centres - normal * thickness])
    try:
        return Delaunay(centres)
    except QhullError:
        return None


def _inside(hull, points: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), 200_000):
        block = points[start : start + 200_000]
        inside[start : start + 200_000] = hull.find_simplex(block) >= 0
    return inside


def within_surface_radius(
    positions: np.ndarray, points: np.ndarray, radius: float
) -> np.ndarray:
    """Mask of Gaussians lying within ``radius`` of any SfM point."""
    from scipy.spatial import cKDTree

    if not len(points):
        return np.zeros(len(positions), dtype=bool)
    tree = cKDTree(points)
    near = np.zeros(len(positions), dtype=bool)
    for start in range(0, len(positions), 200_000):
        block = positions[start : start + 200_000]
        distance, _ = tree.query(block, k=1, distance_upper_bound=radius)
        near[start : start + 200_000] = np.isfinite(distance)
    return near


def hull_voxel_indices(grid: Grid, centres: np.ndarray) -> np.ndarray:
    """
    Every voxel whose centre lies inside the camera hull.

    Enumerated rather than derived from where Gaussians happen to be: a ray
    crosses the empty voxels too, so averaging extinction only over voxels that
    contain Gaussians would divide by the fill factor and overstate the haze.
    """
    hull = _hull_of(centres)
    if hull is None:
        return np.empty(0, dtype=np.int64)
    lo_idx = np.maximum(
        np.floor((centres.min(axis=0) - grid.origin) / grid.voxel).astype(np.int64), 0
    )
    hi_idx = np.minimum(
        np.ceil((centres.max(axis=0) - grid.origin) / grid.voxel).astype(np.int64) + 1,
        grid.dims,
    )
    axes = [np.arange(lo_idx[a], hi_idx[a], dtype=np.int64) for a in range(3)]
    if any(len(a) == 0 for a in axes):
        return np.empty(0, dtype=np.int64)
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    centres_xyz = grid.origin + (mesh + 0.5) * grid.voxel
    keep = _inside(hull, centres_xyz)
    kept = mesh[keep]
    return kept[:, 0] * (grid.dims[1] * grid.dims[2]) + kept[:, 1] * grid.dims[2] + kept[:, 2]


# ------------------------------------------------------------------- report


def require_same_frame(positions: np.ndarray, points: np.ndarray, *, min_inside: float = 0.8) -> None:
    """
    Abort unless the SfM points sit inside the Gaussian cloud.

    Every quantity here is a spatial comparison between two point sets, so a
    frame mismatch does not fail — it silently reports that almost nothing is
    near a surface. Cheap to check, expensive to miss.
    """
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    fraction = float(np.count_nonzero(inside) / max(len(points), 1))
    if fraction < min_inside:
        raise RuntimeError(
            f"Only {fraction:.1%} of SfM points fall inside the Gaussian bounds — "
            "these two are almost certainly in different coordinate frames.\n"
            f"  Gaussians: {lo.round(2).tolist()} .. {hi.round(2).tolist()}\n"
            f"  SfM points: {points.min(axis=0).round(2).tolist()} .. "
            f"{points.max(axis=0).round(2).tolist()}\n"
            "Both must be in COLMAP space; the stage's xformOp is not applied here."
        )


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
    hull_fog = in_hull & is_free
    hull_unsupported = in_hull & ~near_surface
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
            "unsupported_inside": int(np.count_nonzero(hull_unsupported)),
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
