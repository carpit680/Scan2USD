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
