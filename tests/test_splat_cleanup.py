import numpy as np

from scan2usd.reconstruction.splat_cleanup import (
    compute_keep_mask,
    filter_parallel_arrays,
)


def test_spatial_outlier_removal():
    rng = np.random.default_rng(0)
    core = rng.normal(0.0, 0.1, size=(200, 3))
    floaters = np.array(
        [
            [20.0, 0.0, 0.0],
            [-25.0, 5.0, 0.0],
            [0.0, 30.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.vstack([core, floaters])
    opacities = np.ones(len(positions))
    scales = np.full((len(positions), 3), 0.02)

    keep, removed = compute_keep_mask(
        positions, opacities, scales, outlier_std=5.0, min_opacity=0.0
    )
    assert removed["removed_spatial"] >= 3
    assert not keep[-3:].any()
    assert int(keep[:200].sum()) >= 180

    keep_loose, removed_loose = compute_keep_mask(
        positions, opacities, scales, outlier_std=1e6, min_opacity=0.0
    )
    assert removed_loose["removed_spatial"] == 0
    assert keep_loose.all()


def test_opacity_filter():
    positions = np.zeros((10, 3), dtype=np.float64)
    opacities = np.linspace(0.0, 0.1, 10)
    scales = np.full((10, 3), 0.01)
    keep, removed = compute_keep_mask(
        positions, opacities, scales, outlier_std=100.0, min_opacity=0.05
    )
    assert removed["removed_opacity"] == int(np.count_nonzero(opacities < 0.05))
    assert keep.sum() == int(np.count_nonzero(opacities >= 0.05))


def test_scale_filter():
    positions = np.zeros((5, 3), dtype=np.float64)
    opacities = np.ones(5)
    scales = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [1.0, 0.1, 0.1],
            [0.05, 0.05, 0.05],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    keep, removed = compute_keep_mask(
        positions, opacities, scales, outlier_std=100.0, min_opacity=0.0, max_scale=0.5
    )
    assert removed["removed_scale"] == 2
    assert keep.tolist() == [True, True, False, True, False]


def test_filter_parallel_arrays_preserves_sh_packing():
    n = 6
    keep = np.array([True, False, True, True, False, True])
    element_size = 4
    positions = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    opacities = np.arange(n, dtype=np.float64) / 10.0
    scales = np.ones((n, 3), dtype=np.float64)
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    sh = np.arange(n * element_size * 3, dtype=np.float64).reshape(n * element_size, 3)

    out = filter_parallel_arrays(
        keep,
        positions=positions,
        opacities=opacities,
        scales=scales,
        orientations=orientations,
        sh_coeffs=sh,
        sh_element_size=element_size,
    )
    assert out["positions"].shape == (4, 3)
    assert out["opacities"].shape == (4,)
    assert out["sh_coeffs"].shape == (4 * element_size, 3)
    np.testing.assert_array_equal(out["sh_coeffs"][:element_size], sh[:element_size])
    np.testing.assert_array_equal(
        out["sh_coeffs"][element_size : 2 * element_size],
        sh[2 * element_size : 3 * element_size],
    )


def test_splat_cleanup_config_loads():
    from scan2usd.config import ReconstructionConfig

    cfg = ReconstructionConfig.from_dict(
        {
            "splat_cleanup": {
                "enabled": True,
                "outlier_std": 2.5,
                "min_opacity": 0.02,
                "max_scale": 1.5,
            }
        }
    )
    assert cfg.splat_cleanup.outlier_std == 2.5
    assert cfg.splat_cleanup.min_opacity == 0.02
    assert cfg.splat_cleanup.max_scale == 1.5
    params = cfg.splat_cleanup.to_params()
    assert params.outlier_std == 2.5


def _floater_scene():
    """A dense slab of real surface plus the three floater classes we care about."""
    rng = np.random.default_rng(7)
    surface = rng.uniform([-5, -5, 0], [5, 5, 0.2], size=(4000, 3))
    halo = rng.uniform([-40, -40, -40], [40, 40, 40], size=(200, 3))  # outside volume
    isolated = np.array([[0.0, 0.0, 8.0], [1.0, 1.0, 9.0], [-2.0, 2.0, 7.5]])  # in-volume floaters
    pts = np.vstack([surface, halo, isolated])
    opac = np.full(len(pts), 0.8)
    scales = np.full((len(pts), 3), 0.02)
    scales[:5] = 9.0  # a few view-blocking giants inside the surface slab
    bounds = (np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 0.2]))
    return pts, opac, scales, bounds, len(surface)


def test_scale_fraction_removes_view_blocking_giants():
    pts, opac, scales, bounds, _ = _floater_scene()
    keep, info = compute_keep_mask(
        pts, opac, scales,
        outlier_std=0.0, min_opacity=0.0,
        max_scale_frac=0.05, crop_margin=None, min_neighbors=0,
        observed_bounds=bounds,
    )
    assert info["removed_scale"] == 5
    assert not keep[:5].any()


def test_crop_margin_removes_halo_outside_observed_volume():
    pts, opac, scales, bounds, n_surface = _floater_scene()
    keep, info = compute_keep_mask(
        pts, opac, scales,
        outlier_std=0.0, min_opacity=0.0,
        max_scale_frac=None, crop_margin=0.25, min_neighbors=0,
        observed_bounds=bounds,
    )
    # Surface survives; the scattered halo is cut.
    assert keep[:n_surface].mean() > 0.99
    assert info["removed_crop"] > 100


def test_density_filter_removes_isolated_floaters():
    pts, opac, scales, bounds, n_surface = _floater_scene()
    keep, info = compute_keep_mask(
        pts, opac, scales,
        outlier_std=0.0, min_opacity=0.0,
        max_scale_frac=None, crop_margin=None,
        min_neighbors=3, neighbor_radius_frac=0.05,
        observed_bounds=bounds,
    )
    # The three lone Gaussians floating above the slab are gone.
    assert not keep[-3:].any()
    assert keep[:n_surface].mean() > 0.95


def test_filters_together_keep_the_real_surface():
    pts, opac, scales, bounds, n_surface = _floater_scene()
    keep, _ = compute_keep_mask(
        pts, opac, scales,
        outlier_std=0.0, min_opacity=0.01,
        max_scale_frac=0.05, crop_margin=0.25,
        min_neighbors=3, neighbor_radius_frac=0.05,
        observed_bounds=bounds,
    )
    # Everything that is not real surface is gone, and the surface is intact.
    assert not keep[n_surface:].any()
    assert keep[5:n_surface].mean() > 0.95


def _two_cameras_looking_down_z():
    """Two COLMAP cameras at the origin area looking along +Z."""
    from types import SimpleNamespace

    cams = {
        1: SimpleNamespace(
            camera_id=1, model="PINHOLE", width=100, height=100,
            params=np.array([50.0, 50.0, 50.0, 50.0]),
        )
    }
    imgs = {
        "a": SimpleNamespace(qvec=np.array([1.0, 0, 0, 0]), tvec=np.array([0.0, 0, 0]), camera_id=1),
        "b": SimpleNamespace(qvec=np.array([1.0, 0, 0, 0]), tvec=np.array([0.2, 0, 0]), camera_id=1),
    }
    return cams, imgs


def test_view_counts_sees_points_in_front_and_not_behind():
    from scan2usd.reconstruction.splat_cleanup import compute_view_counts

    cams, imgs = _two_cameras_looking_down_z()
    pts = np.array([
        [0.0, 0.0, 5.0],    # dead ahead: both cameras
        [0.0, 0.0, -5.0],   # behind the cameras: nobody
        [100.0, 0.0, 5.0],  # far off to the side, outside the frustum
    ])
    counts = compute_view_counts(pts, cams, imgs)
    assert counts[0] == 2
    assert counts[1] == 0
    assert counts[2] == 0


def test_visibility_filter_drops_unseen_gaussians():
    pts = np.array([[0.0, 0, 5.0], [0.0, 0, -5.0], [0.0, 0, 6.0]])
    opac = np.full(3, 0.9)
    scales = np.full((3, 3), 0.01)
    keep, info = compute_keep_mask(
        pts, opac, scales,
        outlier_std=0.0, min_opacity=0.0,
        min_view_count=2, view_counts=np.array([2, 0, 5]),
        observed_bounds=(np.array([-1.0, -1, -6]), np.array([1.0, 1, 6])),
    )
    assert info["removed_visibility"] == 1
    assert keep.tolist() == [True, False, True]


def test_cleanup_refuses_to_gut_the_scene():
    """A degenerate model must fail loudly, not silently produce an empty scene."""
    from scan2usd.reconstruction.splat_cleanup import SplatCleanupParams

    # 99% of Gaussians at zero opacity: the MCMC-decay failure mode.
    n = 1000
    pts = np.zeros((n, 3))
    opac = np.concatenate([np.full(990, 0.0), np.full(10, 0.9)])
    scales = np.full((n, 3), 0.01)
    keep, info = compute_keep_mask(
        pts, opac, scales, outlier_std=0.0, min_opacity=0.01,
        observed_bounds=(np.array([-1.0]*3), np.array([1.0]*3)),
    )
    assert keep.sum() == 10  # the mask itself is correct
    # The guard lives in the file-level path; check the params carry it.
    assert SplatCleanupParams().min_keep_fraction == 0.05
