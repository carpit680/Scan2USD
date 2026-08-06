"""Free-space carving math in tools/geometry/analyze_splat.py.

The tool's conclusions are all spatial comparisons between two point sets, and
both failure modes it can have are silent: a coordinate-frame mismatch reports
that nothing is near a surface, and averaging extinction only over voxels that
contain Gaussians reports haze inflated by the fill factor. Both produce
plausible numbers, so both are pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_splat",
    Path(__file__).resolve().parents[1] / "tools" / "geometry" / "analyze_splat.py",
)
analyze_splat = importlib.util.module_from_spec(_SPEC)
# @dataclass resolves annotations through sys.modules, so register before exec.
sys.modules[_SPEC.name] = analyze_splat
_SPEC.loader.exec_module(analyze_splat)


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
