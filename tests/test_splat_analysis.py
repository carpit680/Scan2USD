"""Free-space carving math in tools/geometry/analyze_splat.py.

The tool's conclusions are all spatial comparisons between two point sets, and
both failure modes it can have are silent: a coordinate-frame mismatch reports
that nothing is near a surface, and averaging extinction only over voxels that
contain Gaussians reports haze inflated by the fill factor. Both produce
plausible numbers, so both are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from scan2usd.reconstruction import free_space as analyze_splat


def _box_room(rng, n=4000):
    """Points on the walls of a 4x4x3 box, cameras on a ring inside it."""
    wall = rng.uniform([-2, -2, 0], [2, 2, 3], size=(n, 3))
    face = rng.integers(0, 4, size=n)
    wall[face == 0, 0] = -2.0
    wall[face == 1, 0] = 2.0
    wall[face == 2, 1] = -2.0
    wall[face == 3, 1] = 2.0
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    # Slight height variation: a handheld path is never perfectly level, and a
    # perfectly level one is coplanar, which is a separate case (see below).
    cameras = np.column_stack(
        [0.8 * np.cos(angles), 0.8 * np.sin(angles), 1.5 + 0.1 * np.sin(3 * angles)]
    )
    return wall, cameras


def test_frame_mismatch_is_rejected_not_silently_reported():
    positions = np.random.default_rng(0).uniform(-1, 1, size=(500, 3))
    shifted = positions + 100.0
    with pytest.raises(RuntimeError, match="different coordinate frames"):
        analyze_splat.require_same_frame(positions, shifted)
    analyze_splat.require_same_frame(positions, positions[:100])


def test_carving_marks_the_room_interior_free_and_the_walls_occupied():
    rng = np.random.default_rng(1)
    wall, cameras = _box_room(rng)
    centres = {i: c for i, c in enumerate(cameras)}
    # Every wall point is seen by every camera.
    tracks = [np.arange(len(cameras)) for _ in range(len(wall))]

    grid = analyze_splat.build_grid(wall, resolution=48)
    free, occupied = analyze_splat.carve_free_space(
        grid, centres, wall, tracks, max_rays=200_000, stop_margin=3.0 * grid.voxel
    )

    # A point in the middle of the room is empty; a wall point's voxel is not.
    interior = grid.index_of(np.array([[0.0, 0.0, 1.5]]))[0]
    assert free[interior] > 0
    assert occupied[interior] == 0
    assert np.all(occupied[grid.index_of(wall[:200])] > 0)
    # Carving stops short of the surface, so walls keep their own Gaussians.
    assert np.mean(free[grid.index_of(wall[:200])] == 0) > 0.5


def test_hull_voxels_include_empty_space_not_just_occupied_voxels():
    """The denominator for extinction must count voxels a ray crosses."""
    rng = np.random.default_rng(2)
    wall, cameras = _box_room(rng)
    grid = analyze_splat.build_grid(wall, resolution=48)

    hull_voxels = analyze_splat.hull_voxel_indices(grid, cameras)
    # A single Gaussian in the middle of the hull occupies exactly one voxel;
    # the hull itself must be far larger, or extinction is divided by 1.
    assert len(hull_voxels) > 100
    interior = grid.index_of(np.array([[0.0, 0.0, 1.5]]))[0]
    assert interior in set(hull_voxels.tolist())


def test_gaussians_inside_the_camera_ring_are_flagged():
    rng = np.random.default_rng(3)
    _, cameras = _box_room(rng)
    positions = np.array(
        [
            [0.0, 0.0, 1.5],  # centre of the ring
            [10.0, 10.0, 1.5],  # far outside
        ]
    )
    inside = analyze_splat.inside_camera_hull(positions, cameras)
    assert inside[0] and not inside[1]


def test_a_perfectly_level_camera_path_still_produces_a_hull():
    """Panning at a fixed height is coplanar; Qhull refuses it without joggling."""
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    level = np.column_stack(
        [np.cos(angles), np.sin(angles), np.full_like(angles, 1.5)]
    )
    inside = analyze_splat.inside_camera_hull(
        np.array([[0.0, 0.0, 1.5], [10.0, 10.0, 1.5]]), level
    )
    assert inside[0] and not inside[1]


def test_grid_is_fixed_by_the_capture_not_the_model():
    """
    Two models of the same room must get the same grid.

    Sizing the grid to the Gaussians made every fog number incomparable between
    models: the bedroom's raw export spans 574 units because of a few strays,
    giving a 1.36-unit voxel and a 57-voxel camera hull, while the cleaned model
    of the same room gave 0.15 and 48,832. Comparing those measured the grid.
    """
    rng = np.random.default_rng(7)
    wall, _ = _box_room(rng)
    reference = np.percentile(wall, [1, 99], axis=0)

    tight = analyze_splat.build_grid(reference, 64, dilation=0.15)
    # Same capture, but a model carrying a few far-flung stray Gaussians.
    strays = np.vstack([wall, [[500.0, 500.0, 500.0]]])
    still_reference = np.percentile(strays, [1, 99], axis=0)
    loose = analyze_splat.build_grid(still_reference, 64, dilation=0.15)

    assert tight.voxel == pytest.approx(loose.voxel, rel=0.05)
    assert tight.dims.tolist() == loose.dims.tolist()


def test_points_outside_the_grid_are_excluded_not_clamped():
    """A stray Gaussian must not be filed into an edge voxel as if observed."""
    rng = np.random.default_rng(8)
    wall, _ = _box_room(rng)
    grid = analyze_splat.build_grid(np.percentile(wall, [1, 99], axis=0), 48)
    probes = np.array([[0.0, 0.0, 1.5], [900.0, 900.0, 900.0]])
    inside = grid.contains(probes)
    assert inside[0] and not inside[1]


def _carved_room(rng, n=6000):
    """A box room with wall points, an inside camera ring, and its carve."""
    wall, cameras = _box_room(rng, n)
    centres = {i: c for i, c in enumerate(cameras)}
    tracks = [np.arange(len(cameras)) for _ in range(len(wall))]
    reference = analyze_splat.observed_reference_bounds(wall, cameras)
    grid = analyze_splat.build_grid(
        reference, 64, dilation=analyze_splat.OBSERVED_DILATION
    )
    free, occupied = analyze_splat.carve_free_space(
        grid, centres, wall, tracks, max_rays=300_000, stop_margin=3.0 * grid.voxel
    )
    return analyze_splat.CarveResult(
        grid=grid,
        free=free,
        occupied=occupied,
        points=wall,
        centres=cameras,
        reference=reference,
    )


def test_removal_mask_deletes_midair_gaussians_and_spares_wall_ones():
    rng = np.random.default_rng(11)
    carve = _carved_room(rng)
    midair = np.array([[0.0, 0.0, 1.5], [0.3, -0.2, 1.4]])
    on_wall = carve.points[:50]
    positions = np.vstack([midair, on_wall])

    remove, stats = analyze_splat.free_space_removal_mask(
        positions, carve, min_free_votes=3
    )
    assert remove[: len(midair)].all(), "haze in carved-empty air must be removed"
    assert not remove[len(midair) :].any(), "Gaussians on a wall must survive"
    assert stats["free_space_surface_loss"] == 0


def test_more_votes_never_removes_more():
    """Raising the threshold must be monotonically more conservative."""
    rng = np.random.default_rng(12)
    carve = _carved_room(rng)
    probes = np.vstack(
        [np.column_stack([rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300),
                          rng.uniform(1.0, 2.0, 300)]), carve.points[:200]]
    )
    counts = [
        int(analyze_splat.free_space_removal_mask(probes, carve, min_free_votes=v)[0].sum())
        for v in (1, 3, 10, 30)
    ]
    assert counts == sorted(counts, reverse=True)


def _tiny_colmap(tmp_path, seed=0):
    """A minimal binary COLMAP model the carve can actually read."""
    import struct

    rng = np.random.default_rng(seed)
    wall, cameras = _box_room(rng, n=400)
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)

    with open(sparse / "images.bin", "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for index, centre in enumerate(cameras, start=1):
            # Identity rotation, so the stored tvec is -centre.
            fid.write(struct.pack("<i", index))
            fid.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
            fid.write(struct.pack("<ddd", *(-centre)))
            fid.write(struct.pack("<i", 1))
            fid.write(b"f.jpg\x00")
            fid.write(struct.pack("<Q", 0))

    with open(sparse / "points3D.bin", "wb") as fid:
        fid.write(struct.pack("<Q", len(wall)))
        for point in wall:
            fid.write(struct.pack("<Q", 1))
            fid.write(struct.pack("<ddd", *point))
            fid.write(struct.pack("<BBB", 200, 200, 200))
            fid.write(struct.pack("<d", 0.5))
            track = list(range(1, 5))
            fid.write(struct.pack("<Q", len(track)))
            for image_id in track:
                fid.write(struct.pack("<ii", image_id, 0))
    return sparse, wall


def test_carve_cache_returns_the_same_result_and_is_faster(tmp_path):
    """A cache that changes the answer is worse than no cache."""
    sparse, wall = _tiny_colmap(tmp_path)
    positions = wall + np.array([0.0, 0.0, 0.01])
    cache = tmp_path / "carve_cache"

    cold = analyze_splat.carve_from_colmap(
        positions, sparse, resolution=32, max_rays=20_000, cache_dir=cache
    )
    # The exact filename, not a glob: an earlier version left a
    # "carve_<key>.npz.tmp.npz" turd behind when the atomic rename failed, and a
    # glob for carve_*.npz matched it, so this test passed while the cache was
    # never actually written or read.
    key = analyze_splat.carve_cache_key(sparse, 32, 20_000)
    assert (cache / f"carve_{key}.npz").is_file()
    assert not list(cache.glob("*.tmp*")), "temporary files must not survive"

    # Prove the warm run reads the cache rather than recomputing, by making the
    # expensive step fail loudly if it is reached. Hiding the model file would
    # not work: the cache key stats it, so removing it disables the cache too.
    def _must_not_run(*args, **kwargs):
        raise AssertionError("carve_free_space ran on a cache hit")

    original = analyze_splat.carve_free_space
    analyze_splat.carve_free_space = _must_not_run
    try:
        warm = analyze_splat.carve_from_colmap(
            positions, sparse, resolution=32, max_rays=20_000, cache_dir=cache
        )
    finally:
        analyze_splat.carve_free_space = original

    assert np.array_equal(cold.free, warm.free)
    assert np.array_equal(cold.occupied, warm.occupied)
    assert np.array_equal(cold.points, warm.points)
    assert np.allclose(cold.reference, warm.reference)
    assert cold.grid.dims.tolist() == warm.grid.dims.tolist()
    assert cold.grid.voxel == pytest.approx(warm.grid.voxel)


def test_carve_cache_key_tracks_the_inputs_that_change_the_answer(tmp_path):
    """Grid knobs and the COLMAP model must all invalidate; nothing else should."""
    sparse, _ = _tiny_colmap(tmp_path)
    base = analyze_splat.carve_cache_key(sparse, 32, 20_000)

    assert analyze_splat.carve_cache_key(sparse, 64, 20_000) != base
    assert analyze_splat.carve_cache_key(sparse, 32, 40_000) != base
    assert analyze_splat.carve_cache_key(sparse, 32, 20_000) == base

    # Re-running COLMAP rewrites the model; the cache must not survive that.
    import os

    path = sparse / "points3D.bin"
    os.utime(path, (0, 0))
    assert analyze_splat.carve_cache_key(sparse, 32, 20_000) != base


def test_a_corrupt_cache_is_ignored_not_fatal(tmp_path):
    sparse, wall = _tiny_colmap(tmp_path)
    cache = tmp_path / "carve_cache"
    cache.mkdir()
    key = analyze_splat.carve_cache_key(sparse, 32, 20_000)
    (cache / f"carve_{key}.npz").write_bytes(b"not an npz")

    result = analyze_splat.carve_from_colmap(
        wall, sparse, resolution=32, max_rays=20_000, cache_dir=cache
    )
    assert result.free.size > 0
