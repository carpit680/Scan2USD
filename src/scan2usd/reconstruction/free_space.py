"""
Free-space carving: which Gaussians sit in volume the cameras looked *through*.

Every (camera, SfM point) pair is a ray whose interior is known to be empty --
the camera saw that point, so nothing opaque was in the way. Voxels along those
segments are carved free. A Gaussian in carved-free space with no surface nearby
is an artifact by construction rather than by threshold, which is what makes
this usable as a filter: photometric loss provably cannot remove such Gaussians
on its own (their opacity gradients vanish once the blended colour reaches
equilibrium), and held-out PSNR cannot see them because they reproduce roughly
the pixels the surface behind them would.

Shared by ``tools/geometry/analyze_splat.py`` (measurement) and
``reconstruction/splat_cleanup.py`` (removal) so the number that gets reported
and the rule that gets applied can never drift apart.

Everything here is numpy-only except :func:`within_surface_radius`, so it runs
under Isaac's Python as well as the main venv.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The analysis grid spans the observed volume padded by this fraction of its
# extent -- enough to cover wall thickness and the Gaussians just behind surfaces.
OBSERVED_DILATION = 0.25
# A Gaussian within this fraction of the observed diagonal of an SfM point is
# carrying a surface. Resolution-independent, unlike voxel occupancy.
SURFACE_RADIUS_FRAC = 0.015

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




def require_same_frame(
    positions: np.ndarray, points: np.ndarray, *, min_inside: float = 0.8
) -> None:
    """
    Abort unless the SfM points sit inside the Gaussian cloud.

    Every quantity here is a spatial comparison between two point sets, so a
    frame mismatch does not fail -- it silently reports that almost nothing is
    near a surface. Cheap to check, expensive to miss. Authored Gaussian
    positions are in COLMAP space (3DGRUT exports them there and the pipeline
    only ever adds an xformOp on the root stage), so nothing needs transforming;
    this is what makes that a verified fact rather than an assumption.
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


@dataclass(frozen=True)
class CarveResult:
    grid: Grid
    free: np.ndarray
    occupied: np.ndarray
    points: np.ndarray
    centres: np.ndarray
    reference: np.ndarray


def carve_from_colmap(
    positions: np.ndarray,
    sparse_dir: Path,
    *,
    resolution: int = 256,
    max_rays: int = 400_000,
) -> CarveResult:
    """Read a COLMAP sparse model and carve the volume its cameras saw through."""
    sparse_dir = Path(sparse_dir)
    centres_by_id = read_images_bin(sparse_dir / "images.bin")
    points, tracks = read_points3d_bin(sparse_dir / "points3D.bin")
    require_same_frame(positions, points)

    centres = np.asarray(list(centres_by_id.values()), dtype=np.float64)
    reference = observed_reference_bounds(points, centres)
    grid = build_grid(reference, resolution, dilation=OBSERVED_DILATION)
    free, occupied = carve_free_space(
        grid,
        centres_by_id,
        points,
        tracks,
        max_rays=max_rays,
        stop_margin=3.0 * grid.voxel,
    )
    return CarveResult(
        grid=grid,
        free=free,
        occupied=occupied,
        points=points,
        centres=centres,
        reference=reference,
    )


def free_space_removal_mask(
    positions: np.ndarray,
    carve: CarveResult,
    *,
    min_free_votes: int = 3,
    surface_radius_frac: float = SURFACE_RADIUS_FRAC,
) -> tuple[np.ndarray, dict]:
    """
    Mask of Gaussians to delete: carved-free volume, with no surface nearby.

    Both conditions are required. A featureless white wall has few SfM points,
    but there is nothing behind it to aim rays at either, so it fails the
    carved-free test and is never mistaken for haze. The surface radius is
    metric rather than voxel-based so the rule means the same thing at any grid
    resolution.
    """
    positions = np.asarray(positions, dtype=np.float64)
    grid = carve.grid
    in_grid = grid.contains(positions)
    voxel_of = grid.index_of(positions)
    radius = float(surface_radius_frac) * float(
        np.linalg.norm(carve.reference[1] - carve.reference[0])
    )
    near_surface = within_surface_radius(positions, carve.points, radius)
    remove = in_grid & (carve.free[voxel_of] >= int(min_free_votes)) & ~near_surface
    return remove, {
        "removed_free_space": int(np.count_nonzero(remove)),
        "free_space_surface_loss": int(np.count_nonzero(remove & near_surface)),
        "free_space_radius": radius,
        "free_space_voxel": float(grid.voxel),
        "free_space_votes": int(min_free_votes),
    }
